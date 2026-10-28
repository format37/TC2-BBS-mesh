"""The Telegram bridge: invites, both directions of the relay, delivery reactions and the mesh menu.

No threads and no network: a FakeTelegram stands in for the Bot API, the
bridge's workers run inline, and routing packets are fed by hand.
"""
import logging
import os
import re
import time

import pytest
import requests

import command_handlers
import telegram_bridge
from conftest import BBS_NUM, InlineWorker
from db_operations import (
    get_db_connection, get_telegram_contacts, get_telegram_links, get_telegram_user, set_telegram_current, utc_stamp
)
from message_processing import process_message
from telegram_bridge import (
    DELIVERED, NOT_DELIVERED, UNLINKED_WINDOW, Bridge, Deliveries, Delivery, Settings, TelegramApi, TelegramError, text
)
from utils import get_user_state, load_env_file, update_user_state

ALEX, MIKE = 0x11111111, 0x22222222
ALEX_ID, MIKE_ID = '!11111111', '!22222222'
ANNA = {'id': 501, 'is_bot': False, 'first_name': 'Anna', 'username': 'anna', 'language_code': 'en'}
OLGA = {'id': 502, 'is_bot': False, 'first_name': 'Ольга', 'language_code': 'ru'}
RUSSIAN = 'Купи молока и хлеба, вечером зайдём к маме. '
CHAT_MENU = '📱Telegram: {}\nWrite here. [L]ist [I]nvite [P]osition [H]elp e[X]it'
INVITE = 'Chat with me through this bot:\nt.me/bbstestbot?start='


class FakeTelegram:
    """Stands in for TelegramApi: updates to hand out, and a record of every call."""

    def __init__(self):
        self.updates = []
        self.calls = []             # (method, params), in order
        self.fail = {}              # method -> the TelegramError every call raises
        self.last_incoming = None   # the message id of the last message pushed
        self._update_id = 0
        self._message_id = 1000

    def _call(self, method, **params):
        self.calls.append((method, params))
        if method in self.fail:
            raise self.fail[method]

    def get_me(self):
        self._call('getMe')
        return {'id': 42, 'is_bot': True, 'username': 'bbstestbot'}

    def set_my_commands(self, commands, language_code=None):
        self._call('setMyCommands', commands=commands, language_code=language_code)

    def get_updates(self, offset=None, timeout=0):
        self._call('getUpdates', offset=offset, timeout=timeout)
        if offset is not None:
            self.updates = [update for update in self.updates if update['update_id'] >= offset]
        return list(self.updates)

    def send_message(self, chat_id, message_text, reply_markup=None):
        self._call('sendMessage', chat_id=chat_id, text=message_text, reply_markup=reply_markup)

    def send_location(self, chat_id, latitude, longitude):
        self._call('sendLocation', chat_id=chat_id, latitude=latitude, longitude=longitude)

    def set_message_reaction(self, chat_id, message_id, emoji):
        self._call('setMessageReaction', chat_id=chat_id, message_id=message_id, emoji=emoji)

    def answer_callback_query(self, callback_query_id):
        self._call('answerCallbackQuery', callback_query_id=callback_query_id)

    # For the tests

    def push(self, update):
        self._update_id += 1
        self.updates.append({'update_id': self._update_id, **update})

    def message(self, user, message_text=None, chat=None, **extra):
        self._message_id += 1
        self.last_incoming = self._message_id
        message = {'message_id': self._message_id, 'date': 0, 'from': user,
                   'chat': chat or {'id': user['id'], 'type': 'private'}, **extra}
        if message_text is not None:
            message['text'] = message_text
        self.push({'message': message})

    def texts(self, chat_id):
        return [params['text'] for method, params in self.calls if method == 'sendMessage' and params['chat_id'] == chat_id]

    def last(self, method):
        return [params for name, params in self.calls if name == method][-1]

    def count(self, method):
        return sum(1 for name, _ in self.calls if name == method)

    def reactions(self, message_id):
        return [params['emoji'] for method, params in self.calls
                if method == 'setMessageReaction' and params['message_id'] == message_id]


class FakeInterfaceRouting:
    @staticmethod
    def ack(packet_id, sender, reason='NONE'):
        from conftest import FakeInterface
        return FakeInterface.routing(packet_id, sender, reason)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def bridge(iface, monkeypatch, caplog):
    """
    A bridge for the bot @bbstestbot, with the mesh users alex and mike.

    The bridge logs the exceptions of its workers instead of raising them, so
    a test that logged one fails, unless it cleared the log.
    """
    iface.add_node(ALEX, 'alex', 'Alex')
    iface.add_node(MIKE, 'mike', 'Mike')
    bridge = Bridge(FakeTelegram(), iface, Settings(), clock=Clock())
    bridge.username = 'bbstestbot'
    bridge._to_mesh = InlineWorker()
    bridge._to_telegram = InlineWorker()
    bridge.timers = []
    bridge._schedule = lambda delay, function, *args: bridge.timers.append((delay, function, args))
    iface.listeners.append(bridge.on_packet)
    monkeypatch.setattr(command_handlers, '_bridge', bridge)
    yield bridge
    crashes = [record for record in caplog.get_records('call') if record.exc_info]
    assert not crashes, crashes[0].getMessage()


def say(iface, message, num=ALEX):
    """A mesh user sends message to the BBS; what the BBS sent back to them."""
    before = len(iface.texts(num))
    process_message(num, message, iface)
    return iface.texts(num)[before:]


def tg(bridge, message_text=None, user=ANNA, chat=None, **extra):
    """A Telegram user writes to the bot; the bot's answers in their chat."""
    chat_id = (chat or {}).get('id', user['id'])
    before = len(bridge.api.texts(chat_id))
    bridge.api.message(user, message_text, chat, **extra)
    bridge.poll_once()
    return bridge.api.texts(chat_id)[before:]


def invite(iface, name='wife', num=ALEX):
    """The mesh user makes an invite from the Telegram menu; returns its code."""
    say(iface, 'T', num)
    say(iface, 'I', num)
    [reply] = say(iface, name, num)
    return re.search(r'start=(\w+)', reply).group(1)


def link(bridge, iface, name='wife', user=ANNA, num=ALEX):
    """A Telegram user redeems the mesh user's invite; returns the link."""
    tg(bridge, f'/start {invite(iface, name, num)}', user)
    [contact] = [c for c in get_telegram_links(user['id']) if c.node_id == f'!{num:08x}']
    return contact


# Linking by invite

def test_an_invite_links_a_telegram_user_and_both_sides_are_told(bridge, iface):
    assert say(iface, 'T') == ['📱Telegram📱\nNo contacts yet. [I]nvite someone, e[X]it']
    assert say(iface, 'I') == ['Name for the contact? (e.g. wife)']
    [reply] = say(iface, 'wife')
    code = re.search(r'start=(\w+)', reply).group(1)
    # Ready to forward to the Telegram person as it is
    assert reply == f'Chat with me through this bot:\nt.me/bbstestbot?start={code}\nValid 7 days, one use.'
    assert len(code) == 6 and set(code) <= set('ABCDEFGHJKLMNPQRSTUVWXYZ23456789')
    [contact] = get_telegram_contacts(ALEX_ID)
    assert (contact.name, contact.tg_user_id, contact.invite_code) == ('wife', None, code)
    assert contact.invite_expires[:10] == utc_stamp(7 * 86400)[:10]
    assert get_user_state(ALEX) == {'command': 'TELEGRAM_CHAT', 'step': 1, 'contact_id': contact.id}
    assert say(iface, 'Hello') == ["Not sent to Telegram: wife hasn't joined yet"]
    assert bridge.api.count('sendMessage') == 0

    [welcome] = tg(bridge, f'/start {code}')
    assert welcome.startswith("You're connected with alex's radio.\n\nEach message goes to the radio as up to "
                              "3 radio messages, about 580 Latin or 290 Cyrillic characters, and up to 60 radio "
                              f"messages an hour. Text only.\n{DELIVERED} on your message: the radio got it. "
                              f"{NOT_DELIVERED}: not confirmed, send it again later.\n\n/contacts")
    assert '/where — where the radio is' in welcome
    assert iface.texts(ALEX)[-1] == 'wife (Anna @anna) joined your Telegram chat.'
    [contact] = get_telegram_contacts(ALEX_ID)
    assert (contact.tg_user_id, contact.tg_chat_id, contact.tg_name, contact.invite_code) == (501, 501, 'Anna @anna', None)
    assert get_telegram_user(501) == (contact.id, 'english')


def test_a_russian_speaker_gets_the_bot_in_russian(bridge, iface):
    [welcome] = tg(bridge, f'/start {invite(iface)}', OLGA)
    assert welcome.startswith('Вы на связи с радио alex.\n\nКаждое сообщение уходит по радио: до 3 радиосообщений, '
                              'это около 580 латинских или 290 кириллических символов, и не больше 60 '
                              'радиосообщений в час.')
    assert iface.texts(ALEX)[-1] == 'wife (Ольга) joined your Telegram chat.'
    long = (RUSSIAN * 8).strip()
    assert tg(bridge, long, OLGA) == [f'Слишком длинно: {len(long.encode())} байт. В 3 радиосообщения помещается '
                                      'около 580 байт: 580 латинских или 290 кириллических символов.']


def test_an_invite_works_once_and_only_before_it_expires(bridge, iface, caplog):
    code = invite(iface)
    # Typed by hand, in lower case
    assert tg(bridge, f'/start {code.lower()}')[0].startswith("You're connected with alex's radio.")
    sent = len(iface.sent)
    bad = 'Приглашение неизвестно или устарело.'
    with caplog.at_level(logging.INFO):
        assert tg(bridge, f'/start {code}', OLGA) == [bad]
    assert 'user 502 sent an unknown or expired invite code' in caplog.text
    assert tg(bridge, '/start ZZZZZZ', OLGA) == [bad]

    expired = invite(iface, 'sister')
    db = get_db_connection()
    db.execute('UPDATE telegram_contacts SET invite_expires = ? WHERE invite_code = ?', (utc_stamp(-1), expired))
    db.commit()
    assert tg(bridge, f'/start {expired}', OLGA) == [bad]
    assert get_telegram_links(502) == [] and get_telegram_user(502) == (None, None)
    assert len(iface.sent) == sent + 3     # only the menu answers to alex's T, I and name
    # An expired invite leaves the list, and its name is free again
    assert say(iface, 'L') == ['1 wife (Anna @anna)\nNumber to open, D+number to remove, e[X]it']


def test_the_code_alone_works_too(bridge, iface):
    code = invite(iface)
    assert tg(bridge, f' {code.lower()} ')[0].startswith("You're connected with alex's radio.")


def test_a_user_linked_already_keeps_the_link_and_the_new_invite_is_dropped(bridge, iface):
    link(bridge, iface)
    code = invite(iface, 'anna')
    assert tg(bridge, f'/start {code}') == ["You're already connected with alex."]
    assert iface.texts(ALEX)[-1] == 'Invite for anna dropped: Anna @anna is already your contact wife.'
    assert [contact.name for contact in get_telegram_contacts(ALEX_ID)] == ['wife']


def test_nobody_uninvited_reaches_the_radio_or_learns_about_the_mesh(bridge, iface):
    link(bridge, iface)
    sent = len(iface.sent)
    help_ru = text('start_help', 'russian')
    for message_text in ['Привет', '/contacts', '/status', '/where', '/to alex', '/leave alex', '/help', '/start']:
        assert tg(bridge, message_text, OLGA) == [help_ru]
        bridge._clock.now += UNLINKED_WINDOW     # past the limit on answers to unlinked users
    assert tg(bridge, None, OLGA, photo=[{'file_id': 'p'}]) == [help_ru]
    # Groups get nothing, even from a linked user
    group = {'id': -100, 'type': 'group'}
    assert tg(bridge, 'Hello all', ANNA, chat=group) == []
    assert tg(bridge, '/start AAAAAA', OLGA, chat=group) == []
    assert len(iface.sent) == sent
    assert get_telegram_user(502) == (None, None)
    assert bridge.api.count('sendLocation') == 0


def test_contact_names(bridge, iface):
    say(iface, 'T')
    say(iface, 'I')
    too_long = "Name: 1-16 characters, no ':'. Another name, or e[X]it"
    assert say(iface, 'a' * 17) == [too_long]
    assert say(iface, 'my: wife') == [too_long]
    assert say(iface, 'Жена')[0].startswith(INVITE)
    say(iface, 'I')
    assert say(iface, 'жена') == ['You already have жена. Another name, or e[X]it']
    assert say(iface, 'Wife')[0].startswith(INVITE)
    say(iface, 'I')
    assert say(iface, 'WIFE') == ['You already have WIFE. Another name, or e[X]it']
    # Another node may use the same names
    say(iface, 'T', MIKE)
    say(iface, 'I', MIKE)
    assert say(iface, 'жена', MIKE)[0].startswith(INVITE)
    assert [contact.name for contact in get_telegram_contacts(ALEX_ID)] == ['Жена', 'Wife']
    assert [contact.name for contact in get_telegram_contacts(MIKE_ID)] == ['жена']


def test_an_invite_needs_the_bots_name_from_telegram(bridge, iface):
    bridge.username = None      # getMe hasn't succeeded yet
    say(iface, 'T')
    say(iface, 'I')
    assert say(iface, 'wife') == ['Telegram is unreachable now. Try again later.']
    assert get_telegram_contacts(ALEX_ID) == []


# Telegram -> radio

def test_a_telegram_message_reaches_the_radio_after_the_contacts_name(bridge, iface):
    link(bridge, iface)
    assert tg(bridge, ' Buy milk ') == []
    assert iface.texts(ALEX)[-1] == 'wife: Buy milk'


def test_a_long_message_goes_as_numbered_parts(bridge, iface):
    link(bridge, iface)
    before = len(iface.texts(ALEX))
    message_text = (RUSSIAN * 6).strip()
    assert tg(bridge, message_text) == []
    parts = iface.texts(ALEX)[before:]
    assert [part[:4] for part in parts] == ['1/3 ', '2/3 ', '3/3 ']
    assert parts[0].startswith('1/3 wife: Купи') and 'wife' not in parts[1] + parts[2]
    assert ' '.join(part[4:] for part in parts).split() == f'wife: {message_text}'.split()


def test_a_message_over_the_limit_is_refused_and_nothing_is_sent(bridge, iface):
    link(bridge, iface)
    sent = len(iface.sent)
    long = (RUSSIAN * 8).strip()
    assert tg(bridge, long) == [f'Too long: {len(long.encode())} bytes. Up to 3 radio messages fit about 580 bytes: '
                                '580 Latin or 290 Cyrillic characters.']
    assert len(iface.sent) == sent


def test_only_text_reaches_the_radio(bridge, iface):
    link(bridge, iface)
    sent = len(iface.sent)
    assert tg(bridge, None, photo=[{'file_id': 'p'}], caption='Look') == ['Only text reaches the radio.']
    assert tg(bridge, None, sticker={'file_id': 's'}) == ['Only text reaches the radio.']
    assert len(iface.sent) == sent


def test_the_hourly_limit_counts_radio_messages(bridge, iface):
    contact = link(bridge, iface)
    assert bridge._take_airtime(contact.id, 58) == 0
    sent = len(iface.sent)
    # 3 parts would make 61 of 60: refused whole
    assert tg(bridge, 'word ' * 100) == ['Limit: 60 radio messages an hour to alex. Try again in 60 min.']
    assert len(iface.sent) == sent
    assert tg(bridge, 'two ' * 60) == []
    assert len(iface.sent) == sent + 2
    assert tg(bridge, 'one more') == ['Limit: 60 radio messages an hour to alex. Try again in 60 min.']
    bridge._clock.now += 30 * 60
    assert tg(bridge, 'one more') == ['Limit: 60 radio messages an hour to alex. Try again in 30 min.']
    bridge._clock.now += 30 * 60
    assert tg(bridge, 'one more') == []
    assert iface.texts(ALEX)[-1] == 'wife: one more'


def test_several_links_and_none_chosen_ask_whom_to_write(bridge, iface):
    alex = link(bridge, iface)
    mike = link(bridge, iface, 'anna', num=MIKE)
    set_telegram_current(501, None)
    sent = len(iface.sent)
    assert tg(bridge, 'Hello') == ['Choose who gets your messages:']
    assert bridge.api.last('sendMessage')['reply_markup'] == {'inline_keyboard': [
        [{'text': 'alex', 'callback_data': f'to:{alex.id}'}], [{'text': 'mike', 'callback_data': f'to:{mike.id}'}]]}
    assert len(iface.sent) == sent


# Delivery status

def test_delivered_only_once_the_node_acked_every_part(bridge, iface):
    link(bridge, iface)
    first = len(iface.sent)
    # Each part gets the implicit ACK at once (FakeInterface.acks): a neighbour heard it, nothing more
    tg(bridge, 'word ' * 60)
    message_id = bridge.api.last_incoming
    assert len(iface.sent) == first + 2
    assert bridge.api.reactions(message_id) == []
    iface.ack(first + 1, ALEX)
    assert bridge.api.reactions(message_id) == []
    iface.ack(first + 2, ALEX)
    assert bridge.api.reactions(message_id) == [DELIVERED]
    assert bridge.api.last('setMessageReaction') == {'chat_id': 501, 'message_id': message_id, 'emoji': '\U0001F54A'}
    # The window passing changes nothing now
    [(delay, expire, args)] = bridge.timers
    assert delay == 120
    expire(*args)
    assert bridge.api.reactions(message_id) == [DELIVERED]


def test_a_nak_means_not_delivered(bridge, iface, caplog):
    link(bridge, iface)
    first = len(iface.sent)
    with caplog.at_level(logging.INFO):
        tg(bridge, 'Hello')
        iface.ack(first + 1, BBS_NUM, 'MAX_RETRANSMIT')
    assert bridge.api.reactions(bridge.api.last_incoming) == [NOT_DELIVERED]
    assert (f'message {bridge.api.last_incoming} from wife to !11111111 not delivered: no ACK from the node for '
            'part 1/1; NAK MAX_RETRANSMIT for part 1/1') in caplog.text


def test_no_ack_in_time_means_not_delivered_until_a_late_ack(bridge, iface, caplog):
    link(bridge, iface)
    first = len(iface.sent)
    tg(bridge, 'word ' * 60)
    message_id = bridge.api.last_incoming
    iface.ack(first + 1, ALEX)
    [(delay, expire, args)] = bridge.timers
    with caplog.at_level(logging.INFO):
        expire(*args)
    assert bridge.api.reactions(message_id) == [NOT_DELIVERED]
    assert 'not delivered: no ACK from the node for part 2/2' in caplog.text
    iface.ack(first + 2, ALEX)
    assert bridge.api.reactions(message_id) == [NOT_DELIVERED, DELIVERED]


def test_an_ack_that_comes_before_the_message_waits_for_it_counts(bridge, iface):
    link(bridge, iface)
    iface.acked_by = ALEX
    tg(bridge, 'word ' * 60)
    assert bridge.api.reactions(bridge.api.last_incoming) == [DELIVERED]


@pytest.mark.parametrize('failing, sent, problem', [
    (1, [], '2 of 2 parts never reached the radio'),
    (2, ['1/2 wife: word'], '1 of 2 parts never reached the radio'),
])
def test_a_part_that_never_reached_the_radio_ends_the_message(bridge, iface, caplog, failing, sent, problem):
    link(bridge, iface)
    iface.acked_by = ALEX
    original = iface.sendData
    calls = []

    def one_fails(data, destinationId, **kwargs):
        calls.append(data)
        if len(calls) == failing:
            raise RuntimeError('radio gone')
        return original(data, destinationId, **kwargs)

    iface.sendData = one_fails
    before = len(iface.texts(ALEX))
    with caplog.at_level(logging.INFO):
        tg(bridge, 'word ' * 60)
    # The node never gets a part without the ones before it
    assert len(calls) == failing
    assert [part[:len('1/2 wife: word')] for part in iface.texts(ALEX)[before:]] == sent
    assert bridge.api.reactions(bridge.api.last_incoming) == [NOT_DELIVERED]
    assert problem in caplog.text


class DeferredWorker:
    """Keeps what is submitted until run(): a busy worker."""

    def __init__(self):
        self.waiting = []

    def submit(self, fn, *args, **kwargs):
        self.waiting.append((fn, args, kwargs))

    def run(self):
        while self.waiting:
            fn, args, kwargs = self.waiting.pop(0)
            fn(*args, **kwargs)


def test_a_message_waiting_for_the_radio_is_dropped_if_its_contact_is_removed(bridge, iface):
    link(bridge, iface)
    bridge._to_mesh = DeferredWorker()
    tg(bridge, 'Buy milk')
    message_id = bridge.api.last_incoming
    say(iface, 'L')
    say(iface, 'D1')
    assert say(iface, 'Y') == ['wife removed.']
    bridge._to_mesh.run()
    assert 'wife: Buy milk' not in iface.texts(ALEX)
    assert bridge.api.reactions(message_id) == [NOT_DELIVERED]


def test_a_long_delivery_window_still_takes_late_acks():
    changes = []
    clock = Clock()
    deliveries = Deliveries(lambda delivery: changes.append(delivery.state), clock, keep=900 + 600)
    delivery = Delivery([7], 1, ALEX, 501, 1001, 1, 'test')
    deliveries.track(delivery)
    clock.now += 900
    deliveries.expire(delivery)
    clock.now += 300
    deliveries.record(FakeInterfaceRouting.ack(7, ALEX))
    assert changes == ['failed', 'delivered']


# Auto-open

@pytest.mark.parametrize('state', [None, {'command': 'MAIN_MENU', 'step': 1}])
def test_a_delivered_message_opens_the_chat_of_an_idle_user(bridge, iface, state):
    contact = link(bridge, iface)
    update_user_state(ALEX, state)
    iface.acked_by = ALEX
    tg(bridge, 'Buy milk')
    assert get_user_state(ALEX) == {'command': 'TELEGRAM_CHAT', 'step': 1, 'contact_id': contact.id}
    assert say(iface, 'OK, on my way') == []
    assert bridge.api.texts(501)[-1] == 'OK, on my way'


def test_a_busy_user_only_gets_the_message(bridge, iface):
    link(bridge, iface)
    other = link(bridge, iface, 'olga', OLGA)
    iface.acked_by = ALEX
    # In the BBS or Utilities menu, M or S would go to Telegram
    for state in [{'command': 'MENU', 'menu': 'bbs', 'step': 1}, {'command': 'MENU', 'menu': 'utilities', 'step': 1},
                  {'command': 'MAIL', 'step': 2}, {'command': 'AI_CHAT', 'step': 1},
                  {'command': 'TELEGRAM_CHAT', 'step': 1, 'contact_id': other.id}]:
        update_user_state(ALEX, state)
        tg(bridge, 'Buy milk')
        assert iface.texts(ALEX)[-1] == 'wife: Buy milk'
        assert get_user_state(ALEX) == state


def test_an_unconfirmed_message_opens_nothing(bridge, iface):
    link(bridge, iface)
    update_user_state(ALEX, None)
    tg(bridge, 'Buy milk')
    [(_, expire, args)] = bridge.timers
    expire(*args)
    assert get_user_state(ALEX) is None


# Radio -> Telegram

def test_the_mesh_users_message_goes_to_telegram_and_nothing_comes_back(bridge, iface):
    link(bridge, iface)
    assert say(iface, 'Coming home at 7') == []
    assert bridge.api.texts(501)[-1] == 'Coming home at 7'
    assert bridge.api.last('sendMessage')['reply_markup'] is None


@pytest.mark.parametrize('error, reply', [
    (TelegramError(403, 'Forbidden: bot was blocked by the user'), 'Not sent to Telegram: wife blocked the bot'),
    (TelegramError(None, 'Connection refused'), 'Not sent to Telegram: network error, try again'),
    (TelegramError(502, 'Bad Gateway'), 'Not sent to Telegram: network error, try again'),
    (TelegramError(400, 'Bad Request: chat not found'), 'Not sent to Telegram: Telegram error 400'),
])
def test_a_failed_send_is_logged_and_told_on_the_radio(bridge, iface, caplog, error, reply):
    link(bridge, iface)
    bridge.api.fail['sendMessage'] = error
    with caplog.at_level(logging.ERROR):
        assert say(iface, 'Coming home') == [reply]
    assert f"!11111111's message to wife (chat 501) was not sent: {error}" in caplog.text


def test_a_telegram_user_linked_to_two_nodes_switches_between_them(bridge, iface):
    alex = link(bridge, iface)
    mike = link(bridge, iface, 'anna', num=MIKE)     # the newest link is current
    say(iface, 'Coming home', ALEX)
    assert bridge.api.texts(501)[-1] == 'alex: Coming home\n/to alex to reply'
    say(iface, 'Hi Anna', MIKE)
    assert bridge.api.texts(501)[-1] == 'mike: Hi Anna'
    tg(bridge, 'Hi Mike')
    assert iface.texts(MIKE)[-1] == 'anna: Hi Mike'

    assert tg(bridge, '/to ALEX') == ['Writing to alex now.']
    tg(bridge, 'See you')
    assert iface.texts(ALEX)[-1] == 'wife: See you'
    say(iface, 'Coming home', ALEX)
    assert bridge.api.texts(501)[-1] == 'alex: Coming home'

    assert tg(bridge, '/contacts') == ['Your radio contacts. ✓ gets your messages:']
    assert bridge.api.last('sendMessage')['reply_markup'] == {'inline_keyboard': [
        [{'text': '✓ alex', 'callback_data': f'to:{alex.id}'}], [{'text': 'mike', 'callback_data': f'to:{mike.id}'}]]}
    bridge.api.push({'callback_query': {'id': 'q1', 'from': ANNA, 'data': f'to:{mike.id}',
                                        'message': {'message_id': 9, 'chat': {'id': 501, 'type': 'private'}}}})
    bridge.poll_once()
    assert bridge.api.last('answerCallbackQuery') == {'callback_query_id': 'q1'}
    assert bridge.api.texts(501)[-1] == 'Writing to mike now.'
    tg(bridge, 'Hello')
    assert iface.texts(MIKE)[-1] == 'anna: Hello'

    assert tg(bridge, '/to') == ['Send /to and a name: alex, mike']
    assert tg(bridge, '/to bob') == ['No radio contact bob. Yours: alex, mike']
    assert tg(bridge, '/to !11111111') == ['Writing to alex now.']


def test_two_nodes_with_one_short_name_are_told_apart_by_their_ids(bridge, iface):
    iface.nodes[MIKE_ID]['user']['shortName'] = 'alex'
    link(bridge, iface)
    link(bridge, iface, 'anna', num=MIKE)
    tg(bridge, '/contacts')
    buttons = [row[0]['text'] for row in bridge.api.last('sendMessage')['reply_markup']['inline_keyboard']]
    assert buttons == ['alex (!11111111)', '✓ alex (!22222222)']
    say(iface, 'Hi', ALEX)
    assert bridge.api.texts(501)[-1] == 'alex (!11111111): Hi\n/to !11111111 to reply'
    assert tg(bridge, '/to alex') == ['No radio contact alex. Yours: !11111111, !22222222']
    assert tg(bridge, '/to !11111111') == ['Writing to alex (!11111111) now.']


def test_a_removed_link_never_hands_its_messages_to_another_radio(bridge, iface):
    alex = link(bridge, iface)
    mike = link(bridge, iface, 'anna', num=MIKE)
    assert tg(bridge, '/to alex') == ['Writing to alex now.']
    say(iface, 'L', ALEX)
    say(iface, 'D1', ALEX)
    say(iface, 'Y', ALEX)
    assert bridge.api.texts(501)[-1] == 'alex disconnected you.\nChoose who gets your messages:'
    assert bridge.api.last('sendMessage')['reply_markup'] == {'inline_keyboard': [
        [{'text': 'mike', 'callback_data': f'to:{mike.id}'}]]}
    sent = len(iface.sent)
    assert tg(bridge, 'are you still there, alex?') == ['Choose who gets your messages:']
    assert len(iface.sent) == sent
    assert tg(bridge, '/to mike') == ['Writing to mike now.']
    # Leaving the one she writes to works the same way
    link(bridge, iface, 'anna', num=ALEX)       # redeeming makes the new link current
    assert tg(bridge, '/leave alex') == ['Disconnected from alex.\nChoose who gets your messages:']
    # Leaving another one leaves her current link alone
    link(bridge, iface, 'anna', num=ALEX)
    assert tg(bridge, '/to mike') == ['Writing to mike now.']
    assert tg(bridge, '/leave alex') == ['Disconnected from alex.']
    tg(bridge, 'Hi')
    assert iface.texts(MIKE)[-1] == 'anna: Hi'


def test_an_unlinked_user_gets_a_few_answers_then_silence(bridge, iface):
    for _ in range(5):
        assert tg(bridge, 'Hello', OLGA) == [text('start_help', 'russian')]
    assert tg(bridge, 'Hello', OLGA) == []
    assert tg(bridge, '/start ZZZZZZ', OLGA) == []
    # A valid code still works
    assert tg(bridge, f'/start {invite(iface)}', OLGA)[0].startswith('Вы на связи с радио alex.')
    bridge._clock.now += UNLINKED_WINDOW
    assert tg(bridge, 'Hello', ANNA) == [text('start_help', 'english')]


def test_leave_unlinks_and_the_mesh_user_is_told(bridge, iface):
    link(bridge, iface)
    assert tg(bridge, '/leave') == ['Send /leave and a name: alex']
    assert tg(bridge, '/leave alex') == ['Disconnected from alex.']
    assert iface.texts(ALEX)[-1] == 'wife (Anna @anna) left your Telegram chat.'
    assert get_telegram_contacts(ALEX_ID) == [] and get_telegram_user(501) == (None, None)
    assert tg(bridge, 'Hello') == [text('start_help', 'english')]
    # A message alex was writing to her now goes nowhere
    assert say(iface, 'Still there?') == ['Not sent to Telegram: that contact was removed',
                                          '📱Telegram📱\nNo contacts yet. [I]nvite someone, e[X]it']


def test_the_mesh_user_removes_a_contact_and_telegram_is_told(bridge, iface):
    wife = link(bridge, iface)
    invite(iface, 'bob')
    assert say(iface, 'L') == ['1 wife (Anna @anna)\n2 bob (waiting)*\nNumber to open, D+number to remove, e[X]it']
    assert say(iface, '1') == ['Telegram: wife']
    say(iface, 'L')
    assert say(iface, 'd1') == ['Remove wife? Y/N']
    assert say(iface, 'n') == ['Not removed.']
    assert get_user_state(ALEX)['contact_id'] == wife.id
    say(iface, 'L')
    assert say(iface, 'D 1') == ['Remove wife? Y/N']
    assert say(iface, 'Y') == ['wife removed.']
    assert bridge.api.texts(501)[-1] == 'alex disconnected you.'
    assert [contact.name for contact in get_telegram_contacts(ALEX_ID)] == ['bob']
    assert get_user_state(ALEX)['contact_id'] == get_telegram_contacts(ALEX_ID)[0].id
    assert tg(bridge, 'Hello') == [text('start_help', 'english')]
    # A stale button answers without acting
    bridge.api.push({'callback_query': {'id': 'q2', 'from': ANNA, 'data': f'to:{wife.id}',
                                        'message': {'message_id': 9, 'chat': {'id': 501, 'type': 'private'}}}})
    bridge.poll_once()
    assert bridge.api.texts(501)[-1] == text('start_help', 'english')


# /where and /status

def test_where_gives_the_position_unless_the_node_turned_it_off(bridge, iface):
    link(bridge, iface)
    assert tg(bridge, '/where') == ["No position known for alex's radio."]
    iface.nodes[ALEX_ID]['position'] = {'latitudeI': 417151000, 'longitudeI': 447827000, 'latitude': 41.7151,
                                        'longitude': 44.7827, 'time': int(time.time()) - 12 * 60}
    assert tg(bridge, '/where') == ["alex's radio, 12 min ago"]
    assert bridge.api.last('sendLocation') == {'chat_id': 501, 'latitude': 41.7151, 'longitude': 44.7827}
    assert say(iface, 'P') == ['Position for Telegram contacts: off']
    assert tg(bridge, '/where') == ["alex doesn't share the position."]
    assert say(iface, 'p') == ['Position for Telegram contacts: on']
    assert tg(bridge, '/where') == ["alex's radio, 12 min ago"]
    assert tg(bridge, '/where', OLGA) == [text('start_help', 'russian')]
    assert bridge.api.count('sendLocation') == 2
    del iface.nodes[ALEX_ID]['position']['time']
    assert tg(bridge, '/where') == ["alex's radio, time unknown"]


def test_status_says_when_the_bbs_last_heard_the_node(bridge, iface):
    link(bridge, iface)
    assert tg(bridge, '/status') == ["The BBS hasn't heard alex's radio yet."]
    iface.nodes[ALEX_ID].update(lastHeard=int(time.time()) - 3 * 3600, hopsAway=2)
    assert tg(bridge, '/status') == ["alex's radio was last heard 3 h ago, 2 hops away."]
    # meshtastic updates lastHeard on text and node info only; any packet counts
    bridge.on_packet({'from': ALEX, 'hopStart': 7, 'hopLimit': 7, 'decoded': {'portnum': 'TELEMETRY_APP'}}, iface)
    assert tg(bridge, '/status') == ["alex's radio was last heard just now, directly."]
    bridge.on_packet({'from': ALEX, 'hopStart': 7, 'hopLimit': 6, 'decoded': {'portnum': 'POSITION_APP'}}, iface)
    assert tg(bridge, '/status', ) == ["alex's radio was last heard just now, 1 hop away."]
    assert tg(bridge, '/status', OLGA) == [text('start_help', 'russian')]
    bridge.on_packet({'decoded': None}, iface)     # junk is ignored


def test_help(bridge, iface):
    link(bridge, iface)
    [reply] = tg(bridge, '/help')
    assert reply.startswith("Your messages go to alex's radio.\n\nEach message goes to the radio as up to 3")
    assert reply.endswith('/leave NAME — disconnect\n/help — this help')
    assert tg(bridge, '/whatever') == [reply]
    assert tg(bridge, '/start') == [reply]


# The mesh menu

def test_the_telegram_menu(bridge, iface, llm):
    link(bridge, iface)
    assert say(iface, 'T') == [CHAT_MENU.format('wife')]
    assert say(iface, 'h') == [CHAT_MENU.format('wife')]
    assert say(iface, 'l') == ['1 wife (Anna @anna)*\nNumber to open, D+number to remove, e[X]it']
    assert say(iface, 'zz') == ['Not sent. Number to open, D+number to remove, e[X]it']
    assert say(iface, '9') == ['Not sent. Number to open, D+number to remove, e[X]it']
    assert say(iface, 'H') == [CHAT_MENU.format('wife')]
    [menu] = say(iface, 'x')
    assert menu.startswith('💾TC² BBS💾') and '[A]I Chat\n[T]elegram\nE[X]IT' in menu
    assert get_user_state(ALEX) == {'command': 'MAIN_MENU', 'step': 1}
    # The main menu's letters and the quick commands still work inside the chat
    for letter, answer in [('q', '✈️QUICK COMMANDS✈️'), ('b', '📰BBS Menu📰'), ('u', '🛠️Utilities Menu🛠️'),
                           ('a', command_handlers.AI_CHAT_MENU), ('CM', 'You have no new messages.')]:
        say(iface, 'x')
        say(iface, 'T')
        assert get_user_state(ALEX)['command'] == 'TELEGRAM_CHAT'
        assert say(iface, letter)[0].startswith(answer)
    assert bridge.api.texts(501)[-1].startswith("You're connected")     # none of it went to Telegram
    # T after an invite opens the contact used last, not the new invite
    invite(iface, 'bob')
    assert say(iface, 'T') == [CHAT_MENU.format('wife')]
    # Choosing one in the list counts as using it
    olga = link(bridge, iface, 'olga', OLGA)
    say(iface, 'T')
    assert say(iface, 'L')[0].startswith('1 wife (Anna @anna)\n2 bob (waiting)\n3 olga (Ольга)*')
    say(iface, '1')
    assert say(iface, 'x')[0].startswith('💾TC² BBS💾')
    assert say(iface, 'T') == [CHAT_MENU.format('wife')]


def test_t_with_the_bridge_off(iface, monkeypatch):
    monkeypatch.setattr(command_handlers, '_bridge', None)
    iface.add_node(ALEX, 'alex')
    assert say(iface, 'T') == ['Telegram bridge is not configured.']
    assert get_user_state(ALEX) is None
    update_user_state(ALEX, {'command': 'TELEGRAM_CHAT', 'step': 1, 'contact_id': 1})
    assert say(iface, 'Hello') == ['Telegram bridge is not configured.']


# Running

class Stop:
    """A stop event that never sleeps; stop_after polls end the loop."""

    def __init__(self):
        self.waits = []
        self.stopped = False

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.waits.append(seconds)

    def set(self):
        self.stopped = True


def test_the_poll_loop_says_hello_backs_off_quietly_and_recovers(bridge, caplog):
    bridge.username = None
    stop = bridge._stop = Stop()
    script = [TelegramError(None, 'Connection reset'), TelegramError(None, 'Connection reset'),
              TelegramError(409, 'Conflict: terminated by other getUpdates request'), []]

    def get_updates(offset=None, timeout=0):
        step = script.pop(0)
        if not script:
            stop.set()
        if isinstance(step, Exception):
            raise step
        return step

    bridge.api.get_updates = get_updates
    with caplog.at_level(logging.INFO):
        bridge.run()
    assert bridge.username == 'bbstestbot'
    assert [(name, params['language_code']) for name, params in bridge.api.calls if name == 'setMyCommands'] == [
        ('setMyCommands', None), ('setMyCommands', 'ru')]
    assert bridge.api.calls[1][1]['commands'][0] == ('contacts', 'Your radio contacts')
    assert stop.waits == [5, 10, 20]
    assert caplog.text.count('Telegram: unreachable') == 1
    assert 'CONFLICT, another instance polls with this bot token' in caplog.text
    assert 'Telegram: reachable again, polling' in caplog.text
    assert 'Telegram bridge: bot @bbstestbot, polling' in caplog.text


def test_a_rejected_token_stops_the_bridge(bridge, caplog):
    bridge.username = None
    bridge._stop = Stop()
    bridge.api.fail['getMe'] = TelegramError(401, 'Unauthorized')
    bridge.run()
    assert 'the bot token was rejected' in caplog.text
    assert bridge._stop.waits == []


def test_an_update_that_fails_is_logged_and_skipped(bridge, iface, caplog):
    link(bridge, iface)
    bridge.api.push({'message': 'not a message'})
    bridge.api.message(ANNA, 'Still here')
    bridge.poll_once()
    assert f'Telegram: update {bridge.api.updates[0]["update_id"]} failed' in caplog.text
    assert iface.texts(ALEX)[-1] == 'wife: Still here'
    bridge.poll_once()
    assert bridge.api.updates == []     # the next poll confirmed both
    caplog.clear()


def test_a_worker_that_fails_logs_why(bridge, iface, caplog):
    link(bridge, iface)
    bridge.api.fail['setMessageReaction'] = RuntimeError('boom')
    iface.acked_by = ALEX
    tg(bridge, 'Buy milk')
    assert 'Telegram: _react failed' in caplog.text
    caplog.clear()


def test_settings_come_from_the_telegram_section():
    import configparser
    config = configparser.ConfigParser()
    assert Settings.from_config(config) == Settings(3, 60, 7, 120)
    config.read_string('[telegram]\nmax_messages = 2   # radio messages\nmax_per_hour = lots\ndelivery_window = 0\n')
    assert Settings.from_config(config) == Settings(max_messages=2)


# The Bot API client

class FakeResponse:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data

    def json(self):
        return self._data


class FakeSession:
    def __init__(self):
        self.posts = []
        self.response = FakeResponse(200, {'ok': True, 'result': True})
        self.error = None

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json, timeout))
        if self.error:
            raise self.error
        return self.response


def test_the_api_client_posts_json_and_keeps_the_token_out_of_errors():
    session = FakeSession()
    api = TelegramApi('123:SECRET', session=session)
    session.response = FakeResponse(200, {'ok': True, 'result': [{'update_id': 7}]})
    assert api.get_updates(offset=5, timeout=30) == [{'update_id': 7}]
    assert session.posts[-1] == ('https://api.telegram.org/bot123:SECRET/getUpdates',
                                 {'timeout': 30, 'allowed_updates': ['message', 'callback_query'], 'offset': 5}, 40)
    api.set_message_reaction(501, 1001, DELIVERED)
    assert session.posts[-1][1] == {'chat_id': 501, 'message_id': 1001,
                                    'reaction': [{'type': 'emoji', 'emoji': '\U0001F54A'}]}
    assert session.posts[-1][2] == 10

    session.response = FakeResponse(403, {'ok': False, 'error_code': 403,
                                          'description': 'Forbidden: bot was blocked by the user'})
    with pytest.raises(TelegramError) as caught:
        api.send_message(501, 'Hi')
    assert caught.value.status == 403 and 'blocked' in caught.value.description

    session.error = requests.ConnectionError("HTTPSConnectionPool(host='api.telegram.org', port=443): Max retries "
                                             "exceeded with url: /bot123:SECRET/sendMessage")
    with pytest.raises(TelegramError) as caught:
        api.send_message(501, 'Hi')
    assert caught.value.status is None
    assert 'SECRET' not in str(caught.value) and '/bot<token>/sendMessage' in str(caught.value)
    assert caught.value.__suppress_context__


def test_the_env_loader_never_overrides_a_set_variable(tmp_path, monkeypatch):
    env = tmp_path / '.env'
    env.write_text('# Telegram\nTELEGRAM_BOT_TOKEN=from-file\nBBS_TEST_OTHER="quoted"\n\nnot a variable\n')
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'from-environment')
    monkeypatch.delenv('BBS_TEST_OTHER', raising=False)
    assert load_env_file(str(tmp_path / 'missing'), str(env)) == str(env)
    assert os.environ['TELEGRAM_BOT_TOKEN'] == 'from-environment'
    assert os.environ['BBS_TEST_OTHER'] == 'quoted'
    assert load_env_file(str(tmp_path / 'missing')) is None


def test_reactions_are_single_code_points():
    # Telegram rejects the variation-selector forms
    assert [f'U+{ord(c):X}' for c in DELIVERED + NOT_DELIVERED] == ['U+1F54A', 'U+1F648']
    assert telegram_bridge.CODE_ALPHABET.isalnum() and not set('01IO') & set(telegram_bridge.CODE_ALPHABET)


def test_the_bridge_hears_routing_packets_the_way_meshtastic_publishes_them(bridge, iface):
    # As in production: server.py subscribes first, then Bridge.start(); meshtastic publishes ACKs on a subtopic
    from pubsub import pub
    received = []

    def receive_packet(packet, interface):
        received.append(packet)

    pub.subscribe(receive_packet, 'meshtastic.receive')
    pub.subscribe(bridge.on_packet, 'meshtastic.receive')
    try:
        link(bridge, iface)
        iface.listeners.clear()
        first = len(iface.sent)
        tg(bridge, 'Hello')
        pub.sendMessage('meshtastic.receive.routing', packet=iface.routing(first + 1, ALEX), interface=iface)
        assert bridge.api.reactions(bridge.api.last_incoming) == [DELIVERED]
        assert len(received) == 1
    finally:
        pub.unsubscribe(bridge.on_packet, 'meshtastic.receive')
        pub.unsubscribe(receive_packet, 'meshtastic.receive')


def test_the_server_runs_the_bridge_only_with_a_token(monkeypatch, iface):
    import configparser
    import server
    loaded = []
    # The repo's real .env sits next to server.py: never read it here
    monkeypatch.setattr(server, 'load_env_file', lambda *paths: loaded.append(paths))
    monkeypatch.setattr(telegram_bridge.Bridge, 'start', lambda self: None)
    monkeypatch.setattr(command_handlers, '_bridge', None)
    monkeypatch.delenv('TELEGRAM_BOT_TOKEN', raising=False)
    config = configparser.ConfigParser()
    config.read_string('[telegram]\nmax_messages = 2\n')
    assert server.start_telegram_bridge({'config': config}, iface) is None
    assert command_handlers._bridge is None
    assert loaded and loaded[0][0] == '.env' and loaded[0][1].endswith(os.sep + '.env')
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', '123:test')
    bridge = server.start_telegram_bridge({'config': config}, iface)
    assert command_handlers._bridge is bridge and bridge.settings == Settings(max_messages=2)
    assert len(loaded) == 1     # a token in the environment wins over .env
