"""
The Telegram bridge: BBS users chat with people on Telegram through one bot.

A mesh user makes an invite in the BBS menu ([T]elegram, then I) and hands
the link to the Telegram person; redeeming it links that Telegram user to the
node. That is the only way to reach a node from Telegram. A linked user's
messages go to the node as radio DMs, prefixed with the name the mesh user
gave them, and the mesh user's messages in the Telegram menu go back to them.

Threads:
- telegram-poll: long-polls getUpdates, links invites, answers commands on
  Telegram. It never touches the radio.
- telegram-to-mesh worker: everything the bridge sends by radio, one message
  at a time, paced by utils.send_message like every other radio message.
- mesh-to-telegram worker: the mesh side's messages to Telegram and the
  delivery reactions, so an HTTP call never stalls the meshtastic thread.
- meshtastic's publishing thread: on_packet records the radio's routing
  answers and when each node was last heard. It never blocks.

The sender on Telegram learns whether the radio user got a message from a
reaction on it: DELIVERED once the node ACKed every part, NOT_DELIVERED on a
NAK or when delivery_window passes first. A late ACK still turns it into
DELIVERED.

The bot token is the whole bot: it comes from TELEGRAM_BOT_TOKEN and is never logged.
"""

import logging
import math
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, fields
from typing import Optional

import requests
from pubsub import pub

from db_operations import (
    add_telegram_invite, delete_telegram_contact, find_telegram_invite, get_telegram_contact,
    get_telegram_links, get_telegram_share_position, get_telegram_user, redeem_telegram_invite,
    set_telegram_current, set_telegram_language, touch_telegram_contact, utc_stamp
)
from utils import MAX_PAYLOAD_BYTES, change_user_state, send_message, split_message

# Reactions on the Telegram sender's message, both in Telegram's ReactionTypeEmoji
# list as single code points (✅ and ❌ are not: 400 REACTION_INVALID)
DELIVERED = '\U0001F54A'        # 🕊 the node ACKed every part
NOT_DELIVERED = '\U0001F648'    # 🙈 a NAK, or no ACK of every part in time

CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'   # no 0/O, no 1/I
CODE_LENGTH = 6
NAME_MAX = 16                   # a contact's name, as the mesh user calls them
TG_NAME_MAX = 32                # the Telegram first name kept for the mesh user's list

POLL_TIMEOUT = 30               # seconds a getUpdates call waits for news
HTTP_TIMEOUT = 10
BACKOFF_MIN, BACKOFF_MAX = 5, 60
RECENT_SECONDS = 600            # how long routing answers and deliveries are kept, for late ACKs
HOUR = 3600

# Mesh states in which a delivered Telegram message opens its chat: the next message is the reply.
# Not MENU: in the BBS and Utilities menus, letters such as M or S would go to Telegram.
IDLE_COMMANDS = ('MAIN_MENU',)

# Answers an unlinked Telegram user gets per UNLINKED_WINDOW seconds; then the bot is silent to them
UNLINKED_REPLIES = 5
UNLINKED_WINDOW = 600

# What the bridge says on the radio (in English, like the rest of the BBS)
RADIO_JOINED = "{name} ({tg_name}) joined your Telegram chat."
RADIO_LEFT = "{name} ({tg_name}) left your Telegram chat."
RADIO_DROPPED = "Invite for {name} dropped: {tg_name} is already your contact {other}."
RADIO_NOT_SENT = "Not sent to Telegram: {reason}"


# What the bot says, by the Telegram user's language; English is the fallback
TEXTS = {
    'start_help': {
        'english': "This bot connects you with a radio user of a TC² BBS on the Meshtastic mesh. "
                   "To start, open the invite link they gave you, or send /start and the code from it, "
                   "e.g. /start AB7K2M.",
        'russian': "Этот бот связывает вас с пользователем радиосети Meshtastic через TC² BBS. "
                   "Чтобы начать, откройте ссылку-приглашение, которую вам дали, или отправьте /start "
                   "и код из неё, например /start AB7K2M.",
    },
    'bad_invite': {
        'english': "Unknown or expired invite.",
        'russian': "Приглашение неизвестно или устарело.",
    },
    'already_linked': {
        'english': "You're already connected with {node}.",
        'russian': "Вы уже на связи с {node}.",
    },
    'welcome': {
        'english': "You're connected with {node}'s radio.",
        'russian': "Вы на связи с радио {node}.",
    },
    'writing_to': {
        'english': "Your messages go to {node}'s radio.",
        'russian': "Ваши сообщения уходят на радио {node}.",
    },
    'rules': {
        'english': "Each message goes to the radio as up to {max_messages} radio messages, about {latin} Latin "
                   "or {cyrillic} Cyrillic characters, and up to {per_hour} radio messages an hour. Text only.\n"
                   f"{DELIVERED} on your message: the radio got it. {NOT_DELIVERED}: not confirmed, "
                   "send it again later.",
        'russian': "Каждое сообщение уходит по радио: до {max_messages} {up_to_messages}, это около {latin} "
                   "латинских или {cyrillic} кириллических символов, и не больше {per_hour} {hour_messages} "
                   "в час. Только текст.\n"
                   f"{DELIVERED} на вашем сообщении — радио его получило. {NOT_DELIVERED} — не подтверждено, "
                   "отправьте позже ещё раз.",
    },
    'commands': {
        'english': "/contacts — your radio contacts\n/to NAME — write to another one\n"
                   "/status — when the radio was last heard\n/where — where the radio is\n"
                   "/leave NAME — disconnect\n/help — this help",
        'russian': "/contacts — ваши радиоконтакты\n/to ИМЯ — писать другому\n"
                   "/status — когда радио было слышно\n/where — где радио\n"
                   "/leave ИМЯ — отключиться\n/help — эта справка",
    },
    'choose': {
        'english': "Choose who gets your messages:",
        'russian': "Выберите, кому писать:",
    },
    'contacts': {
        'english': "Your radio contacts. ✓ gets your messages:",
        'russian': "Ваши радиоконтакты. ✓ — кому вы пишете:",
    },
    'switched': {
        'english': "Writing to {node} now.",
        'russian': "Теперь вы пишете {node}.",
    },
    'to_usage': {
        'english': "Send /to and a name: {names}",
        'russian': "Отправьте /to и имя: {names}",
    },
    'leave_usage': {
        'english': "Send /leave and a name: {names}",
        'russian': "Отправьте /leave и имя: {names}",
    },
    'unknown_name': {
        'english': "No radio contact {name}. Yours: {names}",
        'russian': "Нет радиоконтакта {name}. Ваши: {names}",
    },
    'gone': {
        'english': "That radio contact is gone.",
        'russian': "Этого радиоконтакта больше нет.",
    },
    'left': {
        'english': "Disconnected from {node}.",
        'russian': "Вы отключились от {node}.",
    },
    'disconnected': {
        'english': "{node} disconnected you.",
        'russian': "{node} отключил(а) вас.",
    },
    'only_text': {
        'english': "Only text reaches the radio.",
        'russian': "По радио доходит только текст.",
    },
    'too_long': {
        'english': "Too long: {size} bytes. Up to {max_messages} radio messages fit about {bytes} bytes: "
                   "{latin} Latin or {cyrillic} Cyrillic characters.",
        'russian': "Слишком длинно: {size} байт. В {max_messages} {in_messages} помещается около {bytes} байт: "
                   "{latin} латинских или {cyrillic} кириллических символов.",
    },
    'rate_limit': {
        'english': "Limit: {per_hour} radio messages an hour to {node}. Try again in {minutes} min.",
        'russian': "Лимит: {per_hour} {hour_messages} в час для {node}. Попробуйте через {minutes} мин.",
    },
    'status': {
        'english': "{node}'s radio was last heard {age}{hops}.",
        'russian': "Радио {node} последний раз слышали {age}{hops}.",
    },
    'status_unknown': {
        'english': "The BBS hasn't heard {node}'s radio yet.",
        'russian': "BBS ещё не слышала радио {node}.",
    },
    'where': {
        'english': "{node}'s radio, {age}",
        'russian': "Радио {node}, {age}",
    },
    'where_none': {
        'english': "No position known for {node}'s radio.",
        'russian': "Местоположение радио {node} неизвестно.",
    },
    'where_off': {
        'english': "{node} doesn't share the position.",
        'russian': "{node} не делится местоположением.",
    },
    'to_reply': {
        'english': "/to {handle} to reply",
        'russian': "/to {handle} — ответить",
    },
}

COMMANDS = {
    'english': [
        ('contacts', 'Your radio contacts'),
        ('to', 'Write to another radio contact: /to NAME'),
        ('status', 'When the radio was last heard'),
        ('where', 'Where the radio is'),
        ('leave', 'Disconnect from a radio contact: /leave NAME'),
        ('help', 'How this works'),
    ],
    'russian': [
        ('contacts', 'Ваши радиоконтакты'),
        ('to', 'Писать другому радиоконтакту: /to ИМЯ'),
        ('status', 'Когда радио было слышно'),
        ('where', 'Где радио'),
        ('leave', 'Отключиться от радиоконтакта: /leave ИМЯ'),
        ('help', 'Как это работает'),
    ],
}


def text(key, language, **values):
    templates = TEXTS[key]
    return templates.get(language, templates['english']).format(**values)


def ru_plural(n, one, few, many):
    """The Russian form of a noun after the number n: 1 сообщение, 3 сообщения, 5 сообщений."""
    n = abs(n) % 100
    if 11 <= n <= 14:
        return many
    n %= 10
    return one if n == 1 else few if 2 <= n <= 4 else many


def ru_genitive(n, singular, plural):
    """After до / не больше: до 1 сообщения, до 3 сообщений."""
    return singular if n % 10 == 1 and n % 100 != 11 else plural


def age_text(seconds, language):
    """How long ago, roughly: 'just now', '12 min ago', '3 h ago', '2 days ago'."""
    russian = language == 'russian'
    minutes = int(max(seconds, 0) // 60)
    if minutes < 1:
        return 'только что' if russian else 'just now'
    if minutes < 60:
        return f'{minutes} мин назад' if russian else f'{minutes} min ago'
    hours = minutes // 60
    if hours < 48:
        return f'{hours} ч назад' if russian else f'{hours} h ago'
    return f'{hours // 24} дн. назад' if russian else f'{hours // 24} days ago'


def hops_text(hops, language):
    """', directly', ', 2 hops away', or nothing when the hops aren't known."""
    if hops is None:
        return ''
    if language == 'russian':
        return ', напрямую' if hops == 0 else f", через {hops} {ru_plural(hops, 'узел', 'узла', 'узлов')}"
    return ', directly' if hops == 0 else f", {hops} hop{'' if hops == 1 else 's'} away"


def language_of(user):
    """The language of the bot's texts for a Telegram user: Russian for ru, else English."""
    return 'russian' if (user.get('language_code') or '').lower().startswith('ru') else 'english'


def telegram_name(user):
    """'Anna @anna': how the mesh user sees who redeemed their invite."""
    name = ' '.join((user.get('first_name') or '').split())[:TG_NAME_MAX] or str(user.get('id'))
    if user.get('username'):
        name += f" @{user['username']}"
    return name


def new_invite_code():
    return ''.join(secrets.choice(CODE_ALPHABET) for _ in range(CODE_LENGTH))


def looks_like_code(value):
    value = value.strip().upper()
    return len(value) == CODE_LENGTH and all(ch in CODE_ALPHABET for ch in value)


def contact_name_problem(name):
    """Why a name can't be a contact's name, or None if it can."""
    if not 1 <= len(name) <= NAME_MAX or ':' in name or '\n' in name:
        return f"Name: 1-{NAME_MAX} characters, no ':'. Another name, or e[X]it"
    return None


def node_num(node_id):
    return int(node_id[1:], 16)


def parse_command(message_text):
    """('to', 'alex') for '/to alex' or '/to@bot alex'; (None, None) for text that isn't a command."""
    if not message_text or not message_text.startswith('/'):
        return None, None
    head, *rest = message_text.split(None, 1)
    return head[1:].split('@')[0].lower(), rest[0].strip() if rest else ''


@dataclass
class Settings:
    """The [telegram] section of config.ini."""
    max_messages: int = 3       # radio messages one Telegram message may take
    max_per_hour: int = 60      # radio messages one contact may cause in a rolling hour
    invite_ttl_days: int = 7
    delivery_window: int = 120  # seconds for the node's ACK of every part

    @classmethod
    def from_config(cls, config):
        """Read the settings; a missing or bad value keeps its default."""
        section = config['telegram'] if config.has_section('telegram') else {}
        values = {}
        for setting in fields(cls):
            raw = section.get(setting.name)
            if raw is None:
                continue
            try:
                value = int(raw.split('#')[0].strip())
                if value < 1:
                    raise ValueError
                values[setting.name] = value
            except ValueError:
                logging.warning(f"Telegram: [telegram] {setting.name} = {raw!r} isn't a positive number, "
                                f"using {setting.default}")
        return cls(**values)


class TelegramError(Exception):
    """A Bot API call failed. status is the HTTP status, or None when Telegram wasn't reached."""

    def __init__(self, status, description):
        super().__init__(f"{status if status is not None else 'network error'}: {description}")
        self.status = status
        self.description = description


class TelegramApi:
    """The few Bot API methods the bridge uses, over HTTPS. Tests replace session or base_url."""

    def __init__(self, token, session=None, base_url='https://api.telegram.org'):
        self._token = token
        self.session = session or requests.Session()
        self.base_url = base_url

    def _redact(self, message):
        # A requests error names the URL, and the URL holds the token
        for secret in (self._token, self._token.partition(':')[2]):
            if secret:
                message = message.replace(secret, '<token>')
        return message

    def call(self, method, params=None, http_timeout=HTTP_TIMEOUT):
        """The method's result; raises TelegramError."""
        try:
            response = self.session.post(f'{self.base_url}/bot{self._token}/{method}', json=params or {},
                                         timeout=http_timeout)
        except requests.RequestException as e:
            raise TelegramError(None, self._redact(str(e))) from None
        try:
            data = response.json()
        except ValueError:
            raise TelegramError(response.status_code, 'the answer is not JSON') from None
        if not data.get('ok'):
            raise TelegramError(data.get('error_code', response.status_code), data.get('description', ''))
        return data.get('result')

    def get_me(self):
        return self.call('getMe')

    def set_my_commands(self, commands, language_code=None):
        params = {'commands': [{'command': name, 'description': about} for name, about in commands]}
        if language_code:
            params['language_code'] = language_code
        return self.call('setMyCommands', params)

    def get_updates(self, offset=None, timeout=POLL_TIMEOUT):
        params = {'timeout': timeout, 'allowed_updates': ['message', 'callback_query']}
        if offset is not None:
            params['offset'] = offset
        return self.call('getUpdates', params, http_timeout=timeout + HTTP_TIMEOUT)

    def send_message(self, chat_id, message_text, reply_markup=None):
        params = {'chat_id': chat_id, 'text': message_text}
        if reply_markup:
            params['reply_markup'] = reply_markup
        return self.call('sendMessage', params)

    def send_location(self, chat_id, latitude, longitude):
        return self.call('sendLocation', {'chat_id': chat_id, 'latitude': latitude, 'longitude': longitude})

    def set_message_reaction(self, chat_id, message_id, emoji):
        return self.call('setMessageReaction', {'chat_id': chat_id, 'message_id': message_id,
                                                'reaction': [{'type': 'emoji', 'emoji': emoji}]})

    def answer_callback_query(self, callback_query_id):
        return self.call('answerCallbackQuery', {'callback_query_id': callback_query_id})


@dataclass
class Delivery:
    """A Telegram message relayed to a node, and the node's ACKs of its parts so far."""
    packet_ids: list            # of the parts handed to the radio, in order
    parts: int                  # parts the message took: fewer ids means some never reached the radio
    destination: int            # the node's number
    chat_id: int
    message_id: int             # the Telegram message the reaction goes on
    contact_id: int
    label: str                  # for the log
    acked: set = field(default_factory=set)
    naks: dict = field(default_factory=dict)    # packet id -> error reason
    state: str = 'waiting'      # then 'delivered', or 'failed', which a late ACK can still turn 'delivered'
    created: float = 0.0
    reaction: Optional[str] = None  # the one on the Telegram message now

    def complete(self):
        return len(self.packet_ids) == self.parts and self.acked.issuperset(self.packet_ids)

    def problem(self):
        """What kept it from being delivered, for the log."""
        problems = []
        if len(self.packet_ids) < self.parts:
            problems.append(f'{self.parts - len(self.packet_ids)} of {self.parts} parts never reached the radio')
        missing = [f'{i}/{self.parts}' for i, packet_id in enumerate(self.packet_ids, 1) if packet_id not in self.acked]
        if missing:
            problems.append(f"no ACK from the node for part {', '.join(missing)}")
        problems += [f'NAK {self.naks[packet_id]} for part {i}/{self.parts}'
                     for i, packet_id in enumerate(self.packet_ids, 1) if packet_id in self.naks]
        return '; '.join(problems)


class Deliveries:
    """
    The radio's answers to the bridge's messages, and the messages waiting for them.

    meshtastic calls a message's onResponse once, on its first answer, and that
    is usually the implicit ACK: the BBS's radio heard a neighbour rebroadcast
    it. Only a routing packet from the node itself says the node got it. So
    every routing packet is kept here for RECENT_SECONDS by the id of the packet
    it answers: the node's ACK can come before its message starts waiting.
    """

    def __init__(self, on_change, clock=time.monotonic, keep=RECENT_SECONDS):
        self._on_change = on_change     # called with a delivery that turned delivered or failed
        self._clock = clock
        self._keep = max(keep, RECENT_SECONDS)     # how long a delivery waits for late ACKs
        self._lock = threading.Lock()
        self._answers = OrderedDict()   # packet id -> [(time, from, reason)]
        self._waiting = {}              # packet id -> Delivery

    def record(self, packet):
        """A routing packet came; on meshtastic's publishing thread."""
        decoded = packet.get('decoded') or {}
        packet_id = decoded.get('requestId')
        if packet_id is None:
            return
        sender = packet.get('from')
        reason = (decoded.get('routing') or {}).get('errorReason', 'NONE')
        now = self._clock()
        with self._lock:
            self._prune(now)
            self._answers.setdefault(packet_id, []).append((now, sender, reason))
            delivery = self._waiting.get(packet_id)
            changed = delivery is not None and self._apply(delivery, packet_id, sender, reason)
        if changed:
            self._on_change(delivery)

    def track(self, delivery):
        """Wait for a delivery's ACKs, counting the answers that came already."""
        now = self._clock()
        with self._lock:
            self._prune(now)
            delivery.created = now
            changed = False
            for packet_id in delivery.packet_ids:
                self._waiting[packet_id] = delivery
                for _, sender, reason in self._answers.get(packet_id, ()):
                    changed = self._apply(delivery, packet_id, sender, reason) or changed
            if delivery.state == 'waiting' and len(delivery.packet_ids) < delivery.parts:
                delivery.state = 'failed'
                changed = True
        if changed:
            self._on_change(delivery)

    def expire(self, delivery):
        """The delivery window passed: a delivery still waiting failed."""
        with self._lock:
            if delivery.state != 'waiting':
                return
            delivery.state = 'failed'
        self._on_change(delivery)

    def _apply(self, delivery, packet_id, sender, reason):
        """Count one answer; True if the delivery turned delivered or failed."""
        if delivery.state == 'delivered':
            return False
        if reason != 'NONE':
            delivery.naks[packet_id] = reason
            if delivery.state == 'waiting':
                delivery.state = 'failed'
                return True
            return False
        if sender != delivery.destination:
            return False    # an implicit ACK: only a neighbour heard it
        delivery.acked.add(packet_id)
        if delivery.complete():
            delivery.state = 'delivered'
            return True
        return False

    def _prune(self, now):
        while self._answers:
            packet_id, answers = next(iter(self._answers.items()))
            if answers[-1][0] > now - RECENT_SECONDS:
                break
            del self._answers[packet_id]
        for packet_id in [pid for pid, d in self._waiting.items() if d.created <= now - self._keep]:
            del self._waiting[packet_id]


class Bridge:
    """The bot, and both directions of the relay. start() runs it; tests drive poll_once() instead."""

    def __init__(self, api, interface, settings=None, clock=time.monotonic):
        self.api = api
        self.interface = interface
        self.settings = settings or Settings()
        self.username = None            # the bot's, from getMe: invites need it
        self.offset = None              # the next update to fetch
        self._clock = clock
        self._stop = threading.Event()
        self._thread = None
        self._to_mesh = ThreadPoolExecutor(max_workers=1, thread_name_prefix='telegram-to-mesh')
        self._to_telegram = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mesh-to-telegram')
        self.deliveries = Deliveries(self._delivery_changed, clock, self.settings.delivery_window + RECENT_SECONDS)
        self._unlinked = {}             # Telegram user id -> deque of when the bot answered them unlinked
        self._airtime = {}              # contact id -> deque of when its radio messages went out
        self._airtime_lock = threading.Lock()
        self._heard = {}                # node number -> (epoch seconds, hops or None) of its last packet

    # Running

    def start(self):
        pub.subscribe(self.on_packet, 'meshtastic.receive')
        self._thread = threading.Thread(target=self.run, name='telegram-poll', daemon=True)
        self._thread.start()

    def stop(self):
        """Stop polling and drop the work still waiting; the poll request ends within its timeout."""
        self._stop.set()
        try:
            pub.unsubscribe(self.on_packet, 'meshtastic.receive')
        except Exception:
            pass
        self._to_mesh.shutdown(wait=False, cancel_futures=True)
        self._to_telegram.shutdown(wait=False, cancel_futures=True)

    def run(self):
        """The poll thread: fetch updates and handle them until stop(); back off while Telegram is unreachable."""
        backoff = 0
        while not self._stop.is_set():
            try:
                if self.username is None:
                    self._hello()
                self.poll_once(POLL_TIMEOUT)
                if backoff:
                    logging.info("Telegram: reachable again, polling")
                backoff = 0
                continue
            except TelegramError as e:
                if e.status in (401, 404):
                    logging.error(f"Telegram: the bot token was rejected ({e}). The bridge is off until "
                                  f"TELEGRAM_BOT_TOKEN is fixed and the BBS restarted.")
                    return
                if e.status == 409:
                    logging.error(f"Telegram: CONFLICT, another instance polls with this bot token, and the two "
                                  f"split the updates: stop one of them. ({e})")
                elif not backoff:
                    logging.warning(f"Telegram: unreachable ({e}); retrying quietly")
            except Exception:
                logging.exception("Telegram: poll error")
            backoff = min(BACKOFF_MAX, backoff * 2 or BACKOFF_MIN)
            self._stop.wait(backoff)

    def _hello(self):
        """Learn the bot's username and set its command menu."""
        me = self.api.get_me()
        self.api.set_my_commands(COMMANDS['english'])
        self.api.set_my_commands(COMMANDS['russian'], language_code='ru')
        self.username = me['username']
        logging.info(f"Telegram bridge: bot @{self.username}, polling")

    def poll_once(self, timeout=0):
        """Fetch the next updates and handle them in order; returns how many came."""
        updates = self.api.get_updates(self.offset, timeout)
        for update in updates:
            # Confirmed by the next getUpdates: a crash can replay one batch, never skip one
            self.offset = update['update_id'] + 1
            try:
                self.handle_update(update)
            except Exception:
                logging.exception(f"Telegram: update {update.get('update_id')} failed")
        return len(updates)

    def on_packet(self, packet, interface):
        """Every packet from the radio, on meshtastic's publishing thread: never blocks or raises."""
        try:
            sender = packet.get('from')
            if sender:
                hop_start = packet.get('hopStart')
                hops = max(hop_start - packet.get('hopLimit', 0), 0) if hop_start else None
                self._heard[sender] = (time.time(), hops)
            if (packet.get('decoded') or {}).get('portnum') == 'ROUTING_APP':
                self.deliveries.record(packet)
        except Exception:
            logging.exception("Telegram: a packet couldn't be recorded")

    def _submit(self, worker, function, *args):
        """Run function on a worker; its exceptions are logged, as a Future would keep them to itself."""
        def run():
            try:
                function(*args)
            except Exception:
                logging.exception(f"Telegram: {function.__name__} failed")
        try:
            worker.submit(run)
        except RuntimeError:
            logging.info(f"Telegram: shutting down, {function.__name__} dropped")

    def _schedule(self, delay, function, *args):
        timer = threading.Timer(delay, function, args)
        timer.daemon = True
        timer.start()

    # Telegram updates

    def handle_update(self, update):
        if 'callback_query' in update:
            self._on_callback(update['callback_query'])
        elif 'message' in update:
            self._on_message(update['message'])

    def _on_message(self, message):
        chat = message.get('chat') or {}
        user = message.get('from') or {}
        if chat.get('type') != 'private' or not user.get('id') or user.get('is_bot'):
            return      # private chats only: groups and channels get nothing
        uid, chat_id = user['id'], chat['id']
        language = language_of(user)
        message_text = message.get('text')
        command, argument = parse_command(message_text)

        if command == 'start' and argument:
            self._redeem(message, argument, language)
            return
        links = get_telegram_links(uid)
        if not links:
            # Nothing about the mesh, and nothing on the radio, for someone no node invited
            if command is None and message_text and looks_like_code(message_text):
                self._redeem(message, message_text, language)
            elif self._may_answer_unlinked(uid):
                self._send(chat_id, text('start_help', language))
            return
        if get_telegram_user(uid)[1] != language:
            set_telegram_language(uid, language)

        if command is not None:
            handler = {
                'contacts': self._cmd_contacts, 'to': self._cmd_to, 'status': self._cmd_status,
                'where': self._cmd_where, 'leave': self._cmd_leave,
            }.get(command, self._cmd_help)
            handler(chat_id, uid, links, argument, language)
        elif message_text is None:
            self._send(chat_id, text('only_text', language))
        else:
            self._relay_to_mesh(message, uid, links, message_text.strip(), language)

    def _redeem(self, message, code, language):
        """Link the sender to the node whose invite code this is."""
        chat_id, user = message['chat']['id'], message['from']
        uid = user['id']
        code = code.strip().upper()
        contact = find_telegram_invite(code)
        if contact is None:
            if self._may_answer_unlinked(uid):
                logging.info(f"Telegram: user {uid} sent an unknown or expired invite code")
                self._send(chat_id, text('bad_invite', language))
            return
        tg_name = telegram_name(user)
        node = self.node_name(contact.node_id)
        linked = [link for link in get_telegram_links(uid) if link.node_id == contact.node_id]
        if linked:
            delete_telegram_contact(contact.id)
            logging.info(f"Telegram: {tg_name} (user {uid}) is already {contact.node_id}'s contact "
                         f"{linked[0].name}; the invite for {contact.name} is dropped")
            self._send(chat_id, text('already_linked', language, node=node))
            self._radio(contact.node_id, RADIO_DROPPED.format(name=contact.name, tg_name=tg_name, other=linked[0].name))
            return
        if not redeem_telegram_invite(contact.id, code, uid, chat_id, tg_name):
            self._send(chat_id, text('bad_invite', language))
            return
        set_telegram_current(uid, contact.id)
        set_telegram_language(uid, language)
        logging.info(f"Telegram: {tg_name} (user {uid}) joined {node} ({contact.node_id}) as {contact.name}")
        contact = get_telegram_contact(contact.id)
        self._send(chat_id, '\n\n'.join([text('welcome', language, node=node), self._rules(contact, language),
                                         text('commands', language)]))
        self._radio(contact.node_id, RADIO_JOINED.format(name=contact.name, tg_name=tg_name))

    def _on_callback(self, query):
        """A press on a /contacts button: switch the radio contact."""
        try:
            self.api.answer_callback_query(query['id'])
        except TelegramError as e:
            logging.warning(f"Telegram: answerCallbackQuery failed: {e}")
        user = query.get('from') or {}
        chat = (query.get('message') or {}).get('chat') or {}
        data = query.get('data') or ''
        if chat.get('type') != 'private' or not user.get('id') or not data.startswith('to:'):
            return
        language = language_of(user)
        links = get_telegram_links(user['id'])
        link = next((link for link in links if str(link.id) == data[3:]), None)
        if link is None:
            if links or self._may_answer_unlinked(user['id']):
                self._send(chat['id'], text('gone' if links else 'start_help', language))
            return
        set_telegram_current(user['id'], link.id)
        self._send(chat['id'], text('switched', language, node=self.label(link, links)))

    # Bot commands of a linked user: (chat_id, uid, links, argument, language)

    def _cmd_help(self, chat_id, uid, links, argument, language):
        current = self._current(uid, links)
        first = (text('writing_to', language, node=self.label(current, links)) if current
                 else text('choose', language).rstrip(':') + ': /contacts')
        self._send(chat_id, '\n\n'.join([first, self._rules(current or links[0], language),
                                         text('commands', language)]))

    def _cmd_contacts(self, chat_id, uid, links, argument, language):
        current = self._current(uid, links)
        self._send(chat_id, text('contacts', language), self._buttons(links, current))

    def _cmd_to(self, chat_id, uid, links, argument, language):
        if not argument:
            self._send(chat_id, text('to_usage', language, names=self._handles(links)))
            return
        link = self._find_link(links, argument)
        if link is None:
            self._send(chat_id, text('unknown_name', language, name=argument, names=self._handles(links)))
            return
        set_telegram_current(uid, link.id)
        self._send(chat_id, text('switched', language, node=self.label(link, links)))

    def _cmd_status(self, chat_id, uid, links, argument, language):
        current = self._current(uid, links)
        if current is None:
            self._choose(chat_id, links, language)
            return
        node = self.label(current, links)
        heard, hops = self.last_heard(current.node_id)
        if heard is None:
            self._send(chat_id, text('status_unknown', language, node=node))
        else:
            self._send(chat_id, text('status', language, node=node, age=age_text(time.time() - heard, language),
                                     hops=hops_text(hops, language)))

    def _cmd_where(self, chat_id, uid, links, argument, language):
        current = self._current(uid, links)
        if current is None:
            self._choose(chat_id, links, language)
            return
        node = self.label(current, links)
        if not get_telegram_share_position(current.node_id):
            self._send(chat_id, text('where_off', language, node=node))
            return
        position = (self.interface.nodes.get(current.node_id) or {}).get('position') or {}
        latitude, longitude = position.get('latitude'), position.get('longitude')
        if latitude is None or longitude is None:
            self._send(chat_id, text('where_none', language, node=node))
            return
        try:
            self.api.send_location(chat_id, latitude, longitude)
        except TelegramError as e:
            logging.warning(f"Telegram: sendLocation to chat {chat_id} failed: {e}")
            return
        fixed = position.get('time')
        age = age_text(time.time() - fixed, language) if fixed else (
            'время неизвестно' if language == 'russian' else 'time unknown')
        self._send(chat_id, text('where', language, node=node, age=age))

    def _cmd_leave(self, chat_id, uid, links, argument, language):
        if not argument:
            self._send(chat_id, text('leave_usage', language, names=self._handles(links)))
            return
        link = self._find_link(links, argument)
        if link is None:
            self._send(chat_id, text('unknown_name', language, name=argument, names=self._handles(links)))
            return
        node = self.label(link, links)
        delete_telegram_contact(link.id)
        logging.info(f"Telegram: {link.tg_name} (user {uid}) left {node} ({link.node_id}), contact {link.name}")
        self._tell_unlinked(uid, chat_id, text('left', language, node=node), language)
        self._radio(link.node_id, RADIO_LEFT.format(name=link.name, tg_name=link.tg_name))

    # Telegram -> radio

    def _relay_to_mesh(self, message, uid, links, message_text, language):
        chat_id = message['chat']['id']
        contact = self._current(uid, links)
        if contact is None:
            self._choose(chat_id, links, language)
            return
        node = self.label(contact, links)
        parts = split_message(f'{contact.name}: {message_text}', numbered=True)
        if len(parts) > self.settings.max_messages:
            size = len(message_text.encode('utf-8'))
            logging.info(f"Telegram: {contact.name} -> {contact.node_id}: {size} bytes refused, "
                         f"{len(parts)} radio messages")
            self._send(chat_id, text('too_long', language, size=size, **self._capacity(contact, language)))
            return
        wait = self._take_airtime(contact.id, len(parts))
        if wait:
            logging.info(f"Telegram: {contact.name} -> {contact.node_id}: refused, over "
                         f"{self.settings.max_per_hour} radio messages an hour")
            self._send(chat_id, text('rate_limit', language, per_hour=self.settings.max_per_hour, node=node,
                                     minutes=wait, **self._capacity(contact, language)))
            return
        touch_telegram_contact(contact.id)
        self._submit(self._to_mesh, self._deliver, contact, parts, chat_id, message['message_id'])

    def _deliver(self, contact, parts, chat_id, message_id):
        """
        On the telegram-to-mesh worker: send the parts to the node, then watch for its ACKs.

        A part that never reaches the radio ends the message: the node would
        get the rest without it. Parts that didn't go out fail it at once.
        """
        destination = node_num(contact.node_id)
        delivery = Delivery([], len(parts), destination, chat_id, message_id, contact.id,
                            f"message {message_id} from {contact.name} to {contact.node_id}")
        try:
            if get_telegram_contact(contact.id) is None:
                logging.info(f"Telegram: {delivery.label} not sent: the contact was removed meanwhile")
                return
            for part in parts:
                packet_ids = send_message(part, destination, self.interface)
                if not packet_ids:
                    break
                delivery.packet_ids += packet_ids
        finally:
            self.deliveries.track(delivery)
            self._schedule(self.settings.delivery_window, self.deliveries.expire, delivery)

    def _delivery_changed(self, delivery):
        """A relayed message got the node's ACK of every part, or failed; on whichever thread saw it."""
        if delivery.state == 'delivered':
            logging.info(f"Telegram: {delivery.label} delivered")
            self._auto_open(delivery.destination, delivery.contact_id)
        else:
            logging.info(f"Telegram: {delivery.label} not delivered: {delivery.problem()}")
        self._submit(self._to_telegram, self._react, delivery)

    def _react(self, delivery):
        """On the mesh-to-telegram worker: the reaction for the delivery's state now, which a late ACK may have changed."""
        emoji = DELIVERED if delivery.state == 'delivered' else NOT_DELIVERED
        if delivery.reaction == emoji:
            return
        delivery.reaction = emoji
        try:
            self.api.set_message_reaction(delivery.chat_id, delivery.message_id, emoji)
        except TelegramError as e:
            logging.warning(f"Telegram: reaction on {delivery.label} failed: {e}")

    def _auto_open(self, destination, contact_id):
        """An idle mesh user's next message answers the contact who just reached them."""
        def change(state):
            if state is None or state.get('command') in IDLE_COMMANDS:
                return {'command': 'TELEGRAM_CHAT', 'step': 1, 'contact_id': contact_id}
            return state
        change_user_state(destination, change)

    def _take_airtime(self, contact_id, parts):
        """Count parts against the contact's hourly limit: 0 if they fit, else minutes until they do."""
        now = self._clock()
        with self._airtime_lock:
            sent = self._airtime.setdefault(contact_id, deque())
            while sent and sent[0] <= now - HOUR:
                sent.popleft()
            over = len(sent) + parts - self.settings.max_per_hour
            if over <= 0:
                sent.extend([now] * parts)
                return 0
            if over > len(sent):
                return 60
            return max(1, math.ceil((sent[over - 1] + HOUR - now) / 60))

    def _capacity(self, contact, language):
        """What max_messages radio messages hold for this contact, after the name and the 1/3 numbers."""
        count = self.settings.max_messages
        numbering = len(f'{count}/{count} ') * count if count > 1 else 0
        room = count * MAX_PAYLOAD_BYTES - len(f'{contact.name}: '.encode('utf-8')) - numbering
        room = room // 10 * 10
        per_hour = self.settings.max_per_hour
        return {
            'max_messages': count, 'bytes': room, 'latin': room, 'cyrillic': room // 2,
            'in_messages': ru_plural(count, 'радиосообщение', 'радиосообщения', 'радиосообщений'),
            'up_to_messages': ru_genitive(count, 'радиосообщения', 'радиосообщений'),
            'hour_messages': ru_plural(per_hour, 'радиосообщение', 'радиосообщения', 'радиосообщений'),
        }

    def _rules(self, contact, language):
        return text('rules', language, per_hour=self.settings.max_per_hour, **self._capacity(contact, language))

    # Radio -> Telegram: called by the mesh side's menu

    def create_invite(self, node_id, name):
        """A waiting contact for name; returns (contact id, code). Raises ValueError if the node has that name."""
        expires = utc_stamp(self.settings.invite_ttl_days * 86400)
        for _ in range(10):
            code = new_invite_code()
            try:
                contact_id = add_telegram_invite(node_id, name, code, expires)
            except sqlite3.IntegrityError as e:
                if 'invite_code' in str(e):
                    continue    # another invite has this code: draw again
                raise ValueError(name) from None
            logging.info(f"Telegram: {node_id} made an invite for {name}")
            return contact_id, code
        raise RuntimeError('no free invite code')

    def invite_link(self, code):
        return f't.me/{self.username}?start={code}'

    def send_to_telegram(self, node_id, contact_id, message_text):
        """The mesh user's message for a contact, sent on the mesh-to-telegram worker."""
        self._submit(self._to_telegram, self._send_to_telegram, node_id, contact_id, message_text)

    def _send_to_telegram(self, node_id, contact_id, message_text):
        contact = get_telegram_contact(contact_id)
        if contact is None or contact.node_id != node_id:
            self._radio(node_id, RADIO_NOT_SENT.format(reason="that contact was removed"))
            return
        if contact.tg_user_id is None:
            self._radio(node_id, RADIO_NOT_SENT.format(reason=f"{contact.name} hasn't joined yet"))
            return
        body = message_text
        links = get_telegram_links(contact.tg_user_id)
        if len(links) > 1:
            current_id, language = get_telegram_user(contact.tg_user_id)
            body = f'{self.label(contact, links)}: {message_text}'
            if current_id != contact.id:
                body += '\n' + text('to_reply', language, handle=self.handle(contact, links))
        try:
            self.api.send_message(contact.tg_chat_id, body)
        except TelegramError as e:
            logging.error(f"Telegram: {node_id}'s message to {contact.name} (chat {contact.tg_chat_id}) "
                          f"was not sent: {e}")
            if e.status == 403:
                reason = f"{contact.name} blocked the bot"
            elif e.status is None or e.status == 429 or e.status >= 500:
                reason = "network error, try again"
            else:
                reason = f"Telegram error {e.status}"
            self._radio(node_id, RADIO_NOT_SENT.format(reason=reason))
        except Exception:
            logging.exception(f"Telegram: {node_id}'s message to {contact.name} failed")
            self._radio(node_id, RADIO_NOT_SENT.format(reason="error, try again"))
        else:
            touch_telegram_contact(contact.id)

    def remove_contact(self, contact):
        """The mesh user removes a contact or invite; a linked Telegram user is told."""
        if contact.tg_user_id is None:
            delete_telegram_contact(contact.id)
            logging.info(f"Telegram: {contact.node_id} removed its invite for {contact.name}")
            return
        language = get_telegram_user(contact.tg_user_id)[1] or 'english'
        node = self.label(contact, get_telegram_links(contact.tg_user_id))
        delete_telegram_contact(contact.id)
        logging.info(f"Telegram: {contact.node_id} removed its contact {contact.name}")
        self._submit(self._to_telegram, self._tell_unlinked, contact.tg_user_id, contact.tg_chat_id,
                     text('disconnected', language, node=node), language)

    # Helpers

    def node_name(self, node_id):
        """A node as Telegram users see it: its short name, else its id."""
        user = (self.interface.nodes.get(node_id) or {}).get('user') or {}
        return user.get('shortName') or node_id

    def label(self, contact, links):
        """The node of a link as its Telegram user sees it: the short name, with the id if two of their links share it."""
        name = self.node_name(contact.node_id)
        same = [link for link in links if self.node_name(link.node_id).casefold() == name.casefold()]
        return f'{name} ({contact.node_id})' if len(same) > 1 else name

    def handle(self, contact, links):
        """What /to takes for a link: the short name, or the node id when the short name is shared."""
        name = self.node_name(contact.node_id)
        return name if self.label(contact, links) == name else contact.node_id

    def _handles(self, links):
        return ', '.join(self.handle(link, links) for link in links)

    def _find_link(self, links, wanted):
        wanted = wanted.strip().casefold()
        for link in links:
            if wanted in (link.node_id.casefold(), link.node_id[1:].casefold(), self.label(link, links).casefold()):
                return link
        named = [link for link in links if self.node_name(link.node_id).casefold() == wanted]
        return named[0] if len(named) == 1 else None

    def _current(self, uid, links):
        """
        The link a Telegram user writes to, or None.

        Redeeming an invite makes its link current, so None means the current
        link was removed: the user chooses the next one rather than writing to
        another radio unawares.
        """
        current_id = get_telegram_user(uid)[0]
        return next((link for link in links if link.id == current_id), None)

    def _tell_unlinked(self, uid, chat_id, message_text, language):
        """After a link is gone: say so, and if the user was writing to it, offer the links left."""
        links = get_telegram_links(uid)
        if links and self._current(uid, links) is None:
            self._send(chat_id, message_text + '\n' + text('choose', language), self._buttons(links, None))
        else:
            self._send(chat_id, message_text)

    def _buttons(self, links, current):
        return {'inline_keyboard': [
            [{'text': ('✓ ' if current is not None and link.id == current.id else '') + self.label(link, links),
              'callback_data': f'to:{link.id}'}]
            for link in links]}

    def _may_answer_unlinked(self, uid):
        """Whether the bot still answers this unlinked user: UNLINKED_REPLIES per UNLINKED_WINDOW, so a flood costs little."""
        now = self._clock()
        if len(self._unlinked) > 1000:
            self._unlinked = {key: times for key, times in self._unlinked.items()
                              if times and times[-1] > now - UNLINKED_WINDOW}
        times = self._unlinked.setdefault(uid, deque())
        while times and times[0] <= now - UNLINKED_WINDOW:
            times.popleft()
        if len(times) >= UNLINKED_REPLIES:
            return False
        times.append(now)
        return True

    def _choose(self, chat_id, links, language):
        self._send(chat_id, text('choose', language), self._buttons(links, None))

    def last_heard(self, node_id):
        """(epoch seconds, hops or None) of the last packet the BBS heard from a node; (None, None) if none.

        meshtastic updates a node's lastHeard on text and node info packets only,
        so the bridge also notes every packet it sees.
        """
        node = self.interface.nodes.get(node_id) or {}
        known, hops = node.get('lastHeard'), node.get('hopsAway')
        heard = self._heard.get(node_num(node_id))
        if heard and (not known or heard[0] >= known):
            return heard[0], heard[1] if heard[1] is not None else hops
        return known, hops

    def _send(self, chat_id, message_text, reply_markup=None):
        """A bot message; a failure is logged, never raised."""
        try:
            self.api.send_message(chat_id, message_text, reply_markup)
            return True
        except TelegramError as e:
            logging.warning(f"Telegram: message to chat {chat_id} failed: {e}")
            return False

    def _radio(self, node_id, message_text):
        """A notice for a node, sent on the telegram-to-mesh worker."""
        self._submit(self._to_mesh, self._send_radio, node_id, message_text)

    def _send_radio(self, node_id, message_text):
        try:
            send_message(message_text, node_num(node_id), self.interface)
        except Exception:
            logging.exception(f"Telegram: notice to {node_id} failed")
