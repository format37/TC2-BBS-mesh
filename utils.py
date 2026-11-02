import logging
import os
import re
import threading
import time
from collections import OrderedDict

from meshtastic import BROADCAST_NUM
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

# A message its destination hasn't ACKed RESEND_AFTER seconds after it went to
# the radio is sent once more, as a new packet. The radio stops retrying as
# soon as it hears a neighbour rebroadcast the message (the implicit ACK), so a
# message that a relay passed on once, and that never reached the node, was
# simply lost. In the street test of 2026-11-02, 9 of the 10 replies the node
# never ACKed had been rebroadcast by the relay OTI31. The node's own ACK came
# at most 30 s after its message went to the radio.
RESEND_AFTER = 45
RESENDS = 1
# The NAKs that say the air lost a message, which sending it again can fix
AIR_FAILURES = frozenset({'MAX_RETRANSMIT', 'TIMEOUT', 'NO_ROUTE', 'GOT_NAK'})
RECENT_SECONDS = 600    # how long sent messages and the routing answers about them are kept
ACKED = 'ACKed'

# The radio rejects payloads over DATA_PAYLOAD_LEN (233) bytes.
MAX_PAYLOAD_BYTES = 200
# A peer BBS parses a sync message only whole, so those may use the radio's full payload.
# They are never sent again (see RESEND_AFTER): a peer would store a bulletin or mail twice.
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


def describe_route(packet):
    """How a received packet got here, for the log: hops, the node that last sent it, the signal."""
    parts = []
    hop_start = packet.get('hopStart')
    if hop_start:
        # meshtastic leaves out fields that are 0, so a missing hopLimit means all hops were used
        hops = hop_start - packet.get('hopLimit', 0)
        parts.append('direct' if hops == 0 else f"{hops} hop{'s' if hops > 1 else ''}")
    if packet.get('relayNode') is not None:
        # The last byte of the node number of whoever transmitted it last: the sender itself when direct
        parts.append(f"relay 0x{packet['relayNode']:02x}")
    if 'rxSnr' in packet:
        parts.append(f"SNR {packet['rxSnr']}")
    if 'rxRssi' in packet:
        parts.append(f"RSSI {packet['rxRssi']}")
    if packet.get('viaMqtt'):
        parts.append('via MQTT')
    return ', '.join(parts) or 'route unknown'


def _answer_text(packet, interface):
    """What a routing answer to a sent message says, for the log."""
    decoded = packet.get('decoded') or {}
    request_id = decoded.get('requestId')
    reason = (decoded.get('routing') or {}).get('errorReason', 'NONE')
    if reason != 'NONE':
        return f"sendID {request_id} was not delivered: {reason}"
    if packet.get('from') == interface.myInfo.my_node_num:
        # The radio heard another node relay it; on_packet reports the node's own ACK if it follows
        return f"sendID {request_id} was relayed (implicit ACK)"
    return f"sendID {request_id} was ACKed by {packet.get('fromId') or packet.get('from')} ({describe_route(packet)})"


def _on_answer(answered, interface):
    """The response handler of a message: the radio answered it. Runs on meshtastic's reader thread, so never raises."""
    def on_response(packet):
        answered.set()
        try:
            logging.info(_answer_text(packet, interface))
        except Exception as e:
            logging.info(f"sendID answer not understood: {e}")
    return on_response


class _Sent:
    """
    The messages sent lately, every routing answer about them, and the resends still to be decided.

    meshtastic calls a message's onResponse once, on its first answer, and that
    is usually the implicit ACK. Only a routing packet from the destination
    itself says it got the message, so on_packet keeps every routing answer
    here, by the id of the packet it answers.
    """

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._messages = OrderedDict()  # packet id -> (time handed, destination, first packet id, reply)
        self._answers = OrderedDict()   # packet id -> [(time, from, reason)]
        self._pending = {}              # packet id -> None while its resend is to be decided, or why not to send it

    def handed(self, packet_id, destination, first, reply):
        now = self._clock()
        with self._lock:
            self._prune(now)
            self._messages[packet_id] = (now, destination, first, reply)

    def answer(self, packet_id, sender, reason):
        """Keep a routing answer; True if it is the destination's first after another answer."""
        now = self._clock()
        with self._lock:
            self._prune(now)
            earlier = self._answers.setdefault(packet_id, [])
            message = self._messages.get(packet_id)
            late = (message is not None and sender == message[1] and earlier
                    and not any(s == sender for _, s, _ in earlier))
            earlier.append((now, sender, reason))
            return late

    def first(self, packet_id):
        """The packet a message first went out as: packet_id itself, unless that was a resend."""
        with self._lock:
            message = self._messages.get(packet_id)
        return message[2] if message else packet_id

    def expect(self, packet_id):
        with self._lock:
            self._pending[packet_id] = None

    def pending(self, packet_id):
        with self._lock:
            return packet_id in self._pending and self._pending[packet_id] is None

    def heard_from(self, sender):
        """sender wrote to the BBS: replies to them not ACKed yet are stale, as they got them or asked again."""
        with self._lock:
            for packet_id, why in self._pending.items():
                message = self._messages.get(packet_id)
                if why is None and message and message[1] == sender and message[3]:
                    self._pending[packet_id] = 'the node wrote since'

    def settle(self, packet_id):
        """Decide the resend of packet_id: (None to send it again, else ACKED or why not; first id; reply)."""
        with self._lock:
            why = self._pending.pop(packet_id, None)
            if packet_id not in self._messages:
                return 'forgotten', packet_id, True
            _, destination, first, reply = self._messages[packet_id]
            copies = [pid for pid, message in self._messages.items() if message[2] == first]
            answers = [answer for pid in copies for answer in self._answers.get(pid, ())]
        if any(sender == destination for _, sender, _ in answers):
            return ACKED, first, reply
        refusals = [reason for _, _, reason in answers if reason != 'NONE' and reason not in AIR_FAILURES]
        return why or (refusals[-1] if refusals else None), first, reply

    def _prune(self, now):
        for kept in (self._messages, self._answers):
            while kept:
                packet_id, value = next(iter(kept.items()))
                last = value[-1][0] if kept is self._answers else value[0]
                if last > now - RECENT_SECONDS:
                    break
                del kept[packet_id]
                if kept is self._messages:
                    self._pending.pop(packet_id, None)


_sent = _Sent()


def on_packet(packet, interface):
    """Every packet from the radio, as a meshtastic.receive subscriber: keeps the routing answers. Never raises."""
    try:
        decoded = packet.get('decoded') or {}
        if decoded.get('portnum') != 'ROUTING_APP' or decoded.get('requestId') is None:
            return
        reason = (decoded.get('routing') or {}).get('errorReason', 'NONE')
        if _sent.answer(decoded['requestId'], packet.get('from'), reason):
            logging.info(_answer_text(packet, interface))
    except Exception as e:
        logging.info(f"Routing answer not understood: {e}")


def heard_from(sender):
    """A direct message from node number sender came: the replies to it that it hasn't ACKed won't be sent again."""
    _sent.heard_from(sender)


def first_sent_as(packet_id):
    """The id of the packet a message first went out as, for a resend's id."""
    return _sent.first(packet_id)


def resend_pending(packet_id):
    """True while the BBS may still send the message of packet_id again."""
    return _sent.pending(packet_id)


def _node_num(destination):
    """A destination as routing answers name it: from an int, '!hex' or '^all'; None if it is something else."""
    if isinstance(destination, int):
        return destination
    if destination == '^all':
        return BROADCAST_NUM
    if isinstance(destination, str) and destination.startswith('!'):
        try:
            return int(destination[1:], 16)
        except ValueError:
            return None
    return None


def _schedule(delay, function, *args):
    timer = threading.Timer(delay, function, args)
    timer.daemon = True
    timer.start()


def _hand(chunk, destination, interface, first=None, reply=True):
    """Hand one radio message to the radio, once it answered the previous one; its packet id, or None."""
    global _answered, _handed_at
    packet_id = None
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
                onResponse=_on_answer(answered, interface),
                onResponseAckPermitted=True,
            )
            _answered, _handed_at = answered, time.monotonic()
            packet_id = d.id
            _sent.handed(packet_id, _node_num(destination), first or packet_id, reply)
        destid = get_node_id_from_num(destination, interface)
        size = len(chunk.encode('utf-8'))
        again = f" again (first sendID {first})" if first else ''
        chunk = chunk.replace('\n', '\\n')
        logging.info(f"Sending message{again} to user '{get_node_short_name(destid, interface)}' ({destid}) with sendID {packet_id} ({size} bytes): \"{chunk}\"")
    except Exception as e:
        logging.info(f"REPLY SEND ERROR {e}")
    return packet_id


def _expect_ack(chunk, destination, interface, packet_id, resends):
    """Send chunk again if its destination doesn't ACK packet_id in RESEND_AFTER seconds, up to resends times."""
    num = _node_num(destination)
    if resends < 1 or num is None or num == BROADCAST_NUM:
        return
    _sent.expect(packet_id)
    _schedule(RESEND_AFTER, _resend_unacked, chunk, destination, interface, packet_id, resends)


def _resend_unacked(chunk, destination, interface, packet_id, resends):
    """On a timer thread: send chunk again, unless its destination ACKed or refused it, or wrote since a reply."""
    try:
        why, first, reply = _sent.settle(packet_id)
        if why == ACKED:
            return
        num = _node_num(destination)
        destid = get_node_id_from_num(num, interface) or f'!{num:08x}'
        if why:
            logging.info(f"sendID {packet_id} got no ACK from {destid}, not sending it again: {why}")
            return
        logging.info(f"sendID {packet_id} got no ACK from {destid} in {RESEND_AFTER} s, sending it again")
        resent = _hand(chunk, destination, interface, first, reply)
        if resent is not None:
            _expect_ack(chunk, destination, interface, resent, resends - 1)
    except Exception:
        logging.exception(f"sendID {packet_id}: deciding its resend failed")


def send_message(message, destination, interface, numbered=False, max_bytes=MAX_PAYLOAD_BYTES,
                 resend=True, reply=True):
    """
    Send message as radio messages of at most max_bytes; returns the ids of the packets handed to the radio.

    With resend, a part its destination doesn't ACK in RESEND_AFTER seconds is
    sent once more (broadcasts never are). reply means the message answers the
    destination's last message: then a newer message from it makes the part
    stale, and it isn't sent again. Messages nobody asked for, such as a
    Telegram relay or a mail notification, pass reply=False.
    """
    ids = []
    for chunk in split_message(message, max_bytes, numbered):
        packet_id = _hand(chunk, destination, interface, reply=reply)
        if packet_id is not None:
            ids.append(packet_id)
            if resend:
                _expect_ack(chunk, destination, interface, packet_id, RESENDS)
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
        send_message(message, node_id, interface, max_bytes=SYNC_PAYLOAD_BYTES, resend=False)


def send_mail_to_bbs_nodes(sender_id, sender_short_name, recipient_id, subject, content, unique_id, bbs_nodes,
                           interface):
    message = f"MAIL|{sender_id}|{sender_short_name}|{recipient_id}|{subject}|{content}|{unique_id}"
    logging.info(f"SERVER SYNC: Syncing new mail message {subject} sent from {sender_short_name} to other BBS systems.")
    for node_id in bbs_nodes:
        send_message(message, node_id, interface, max_bytes=SYNC_PAYLOAD_BYTES, resend=False)


def send_delete_bulletin_to_bbs_nodes(unique_id, bbs_nodes, interface):
    message = f"DELETE_BULLETIN|{unique_id}"
    for node_id in bbs_nodes:
        send_message(message, node_id, interface, max_bytes=SYNC_PAYLOAD_BYTES, resend=False)


def send_delete_mail_to_bbs_nodes(unique_id, bbs_nodes, interface):
    message = f"DELETE_MAIL|{unique_id}"
    logging.info(f"SERVER SYNC: Sending delete mail sync message with unique_id: {unique_id}")
    for node_id in bbs_nodes:
        send_message(message, node_id, interface, max_bytes=SYNC_PAYLOAD_BYTES, resend=False)


def send_channel_to_bbs_nodes(name, url, bbs_nodes, interface):
    message = f"CHANNEL|{name}|{url}"
    for node_id in bbs_nodes:
        send_message(message, node_id, interface, max_bytes=SYNC_PAYLOAD_BYTES, resend=False)
