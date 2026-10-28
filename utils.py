import logging
import os
import re
import threading
import time

from meshtastic.protobuf import portnums_pb2

user_states = {}
# The AI chat worker changes states too; change_user_state keeps its read-modify-write whole.
_states_lock = threading.Lock()

# MeshInterface._sendToRadio changes its queue without a lock, and AI chat
# replies are sent from a worker thread, so sends take turns.
_send_lock = threading.Lock()

# A message goes to the radio only once the radio has answered the previous
# one (its ACK, or a NAK), or after ACK_TIMEOUT seconds. Radios on the
# ESP32-S3's own USB port (Heltec V4, T-Beam S3) can leave their serial input
# unread for up to 20 s on firmware before 2.8.1, and a second message that
# arrives meanwhile overflows their 256-byte buffer and is lost
# (meshtastic/firmware#11864). So ACK_TIMEOUT stays above those 20 s. Waiting
# for the ACK also keeps the parts of a reply apart on air.
ACK_TIMEOUT = 30
_answered = threading.Event()
_answered.set()
_handed_at = 0.0

# The radio rejects payloads over DATA_PAYLOAD_LEN (233) bytes.
MAX_PAYLOAD_BYTES = 200
# A peer BBS parses a sync message only whole, so those may use the radio's full payload
SYNC_PAYLOAD_BYTES = 233

_SENTENCE_END = re.compile(r'[.!?…](?=\s)')


def update_user_state(user_id, state):
    with _states_lock:
        user_states[user_id] = state


def get_user_state(user_id):
    return user_states.get(user_id, None)


def change_user_state(user_id, change):
    """Replace a user's state with change(state), atomically against other threads."""
    with _states_lock:
        user_states[user_id] = change(user_states.get(user_id))


def _cut_point(text, limit, ends=True):
    """Where to end a part of text when its first limit characters fit.

    With ends, after the last line or sentence end in the second half of the
    part; else at its last space, else at the limit. Looks one character past
    the limit, so a part that ends right before a space ends there.
    """
    head = text[:limit + 1]
    if ends:
        line = head.rfind('\n')
        if line >= limit // 2:
            return line
        sentence_ends = [m.end() for m in _SENTENCE_END.finditer(head)]
        if sentence_ends and sentence_ends[-1] >= limit // 2:
            return sentence_ends[-1]
    for i in range(limit, 0, -1):
        if head[i].isspace():
            return i
    return limit


def _split(text, max_bytes):
    """The parts of text, cut at line and sentence ends unless that takes more parts than cutting at spaces."""
    parts = _split_at(text, max_bytes, ends=True)
    if len(parts) > 1:
        compact = _split_at(text, max_bytes, ends=False)
        if len(compact) < len(parts):
            return compact
    return parts


def _split_at(text, max_bytes, ends):
    parts = []
    while len(text.encode('utf-8')) > max_bytes:
        # The longest prefix that fits, never ending inside a character
        limit = len(text.encode('utf-8')[:max_bytes].decode('utf-8', 'ignore'))
        cut = _cut_point(text, limit, ends)
        part = text[:cut].rstrip()
        if part:
            parts.append(part)
        text = text[cut:].lstrip()
    if text:
        parts.append(text)
    return parts


def split_message(text, max_bytes=MAX_PAYLOAD_BYTES, numbered=False):
    """Split text into radio messages of at most max_bytes UTF-8 bytes.

    numbered prefixes the parts with "1/3 " when there is more than one.
    """
    parts = _split(text, max_bytes)
    if not numbered or len(parts) < 2:
        return parts
    reserve = 0
    while len(f"{len(parts)}/{len(parts)} ") > reserve:
        reserve = len(f"{len(parts)}/{len(parts)} ")
        parts = _split(text, max_bytes - reserve)
    if len(parts) < 2:
        return parts
    return [f"{i}/{len(parts)} {part}" for i, part in enumerate(parts, 1)]


def truncate(text, max_bytes, ending='...'):
    """text if it fits in max_bytes UTF-8 bytes, else cut at a word boundary and ended with ending."""
    if len(text.encode('utf-8')) <= max_bytes:
        return text
    room = max(max_bytes - len(ending.encode('utf-8')), 0)
    limit = len(text.encode('utf-8')[:room].decode('utf-8', 'ignore'))
    return text[:_cut_point(text, limit)].rstrip(' \t\n,.;:-–—') + ending


def _on_answer(answered):
    """The response handler of a message: the radio answered it. Runs on meshtastic's reader thread, so never raises."""
    def on_response(packet):
        answered.set()
        decoded = packet.get('decoded') or {}
        reason = (decoded.get('routing') or {}).get('errorReason', 'NONE')
        if reason != 'NONE':
            logging.info(f"sendID {decoded.get('requestId')} was not delivered: {reason}")
    return on_response


def send_message(message, destination, interface, numbered=False, max_bytes=MAX_PAYLOAD_BYTES):
    """Send message as radio messages of at most max_bytes; returns the ids of the packets handed to the radio."""
    global _answered, _handed_at
    ids = []
    for chunk in split_message(message, max_bytes, numbered):
        try:
            with _send_lock:
                wait = _handed_at + ACK_TIMEOUT - time.monotonic()
                if wait > 0 and not _answered.wait(wait):
                    logging.info(f"No ACK in {ACK_TIMEOUT} s, sending the next message anyway")
                answered = threading.Event()
                d = interface.sendData(
                    chunk.encode('utf-8'),
                    destinationId=destination,
                    portNum=portnums_pb2.PortNum.TEXT_MESSAGE_APP,
                    wantAck=True,
                    onResponse=_on_answer(answered),
                    onResponseAckPermitted=True,
                )
                _answered, _handed_at = answered, time.monotonic()
                ids.append(d.id)
            destid = get_node_id_from_num(destination, interface)
            size = len(chunk.encode('utf-8'))
            chunk = chunk.replace('\n', '\\n')
            logging.info(f"Sending message to user '{get_node_short_name(destid, interface)}' ({destid}) with sendID {d.id} ({size} bytes): \"{chunk}\"")
        except Exception as e:
            logging.info(f"REPLY SEND ERROR {e}")
    return ids


def load_env_file(*paths):
    """
    Read KEY=value lines from the first of paths that exists into os.environ.

    A variable that is already set keeps its value. Returns the path read, or None.
    """
    for path in paths:
        if not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8') as f:
            for line in f:
                key, sep, value = line.strip().partition('=')
                if sep and key and not key.startswith('#'):
                    os.environ.setdefault(key.strip(), value.strip().strip('\'"'))
        return path
    return None


def get_node_info(interface, short_name):
    nodes = [{'num': node_id, 'shortName': node['user']['shortName'], 'longName': node['user']['longName']}
             for node_id, node in interface.nodes.items()
             if node['user']['shortName'].lower() == short_name]
    return nodes


def get_node_id_from_num(node_num, interface):
    # A copy: the meshtastic reader thread adds nodes while the AI worker reads them
    for node_id, node in list(interface.nodes.items()):
        if node['num'] == node_num:
            return node_id
    return None


def get_node_short_name(node_id, interface):
    node_info = interface.nodes.get(node_id)
    if node_info:
        return node_info['user']['shortName']
    return None


def send_bulletin_to_bbs_nodes(board, sender_short_name, subject, content, unique_id, bbs_nodes, interface):
    message = f"BULLETIN|{board}|{sender_short_name}|{subject}|{content}|{unique_id}"
    for node_id in bbs_nodes:
        send_message(message, node_id, interface, max_bytes=SYNC_PAYLOAD_BYTES)


def send_mail_to_bbs_nodes(sender_id, sender_short_name, recipient_id, subject, content, unique_id, bbs_nodes,
                           interface):
    message = f"MAIL|{sender_id}|{sender_short_name}|{recipient_id}|{subject}|{content}|{unique_id}"
    logging.info(f"SERVER SYNC: Syncing new mail message {subject} sent from {sender_short_name} to other BBS systems.")
    for node_id in bbs_nodes:
        send_message(message, node_id, interface, max_bytes=SYNC_PAYLOAD_BYTES)


def send_delete_bulletin_to_bbs_nodes(unique_id, bbs_nodes, interface):
    message = f"DELETE_BULLETIN|{unique_id}"
    for node_id in bbs_nodes:
        send_message(message, node_id, interface, max_bytes=SYNC_PAYLOAD_BYTES)


def send_delete_mail_to_bbs_nodes(unique_id, bbs_nodes, interface):
    message = f"DELETE_MAIL|{unique_id}"
    logging.info(f"SERVER SYNC: Sending delete mail sync message with unique_id: {unique_id}")
    for node_id in bbs_nodes:
        send_message(message, node_id, interface, max_bytes=SYNC_PAYLOAD_BYTES)


def send_channel_to_bbs_nodes(name, url, bbs_nodes, interface):
    message = f"CHANNEL|{name}|{url}"
    for node_id in bbs_nodes:
        send_message(message, node_id, interface, max_bytes=SYNC_PAYLOAD_BYTES)
