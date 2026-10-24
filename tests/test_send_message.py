"""Radio messages are cut by UTF-8 bytes: 200 characters of Russian or
Georgian are 400-600 bytes, and the radio rejects payloads over 233."""
import logging
import threading
import time

import pytest

from conftest import FakeInterface
from utils import send_bulletin_to_bbs_nodes, send_message, split_message


def sizes(parts):
    return [len(part.encode('utf-8')) for part in parts]


def test_a_short_text_is_one_message_as_it_is():
    assert split_message('Menu\n[R]ead\n') == ['Menu\n[R]ead\n']
    assert split_message('') == []


@pytest.mark.parametrize('word', ['hello', 'привет', 'გამარჯობა'])
def test_a_long_text_is_cut_at_spaces_and_every_part_fits(word):
    text = ' '.join(f'{word}{i}' for i in range(200))
    parts = split_message(text)
    assert len(parts) > 1
    assert max(sizes(parts)) <= 200
    assert ' '.join(parts) == text


def test_a_cut_never_falls_inside_a_character():
    parts = split_message('ა' * 500)
    assert ''.join(parts) == 'ა' * 500
    assert sizes(parts) == [198] * 7 + [114]
    parts = split_message('x' + '😀' * 60)
    assert sizes(parts) == [197, 44]


def test_a_cut_prefers_a_sentence_end_then_a_line_end():
    parts = split_message('The quick brown fox jumps. ' * 10)
    assert parts[0] == ('The quick brown fox jumps. ' * 7).strip()
    parts = split_message('a ' * 60 + '\n' + 'b ' * 59 + 'b')
    assert parts == [('a ' * 60).strip(), ('b ' * 60).strip()]
    # A line end early in the message would waste most of it
    parts = split_message('Subject: hi\n' + 'word ' * 60)
    assert parts[0].startswith('Subject: hi\nword word')


def test_nice_cuts_never_cost_an_extra_message():
    # A reply from the live test: cut at sentence ends it took 3 messages
    text = ('ქსელში 250 კვანძია, ბოლო 24 საათში 33, ბოლო 8 საათში 22 და ბოლო 1 საათში 19. '
            'ბოლო მომხმარებლები: alex, tcsl, Owl. 20%-ზე ნაკლები ბატარეით რამდენიმე კვანძია. დახმარება?')
    parts = split_message(text, numbered=True)
    assert [part[:4] for part in parts] == ['1/2 ', '2/2 ']
    assert max(sizes(parts)) <= 200
    assert ' '.join(part[4:] for part in parts) == text


def test_numbered_parts_count_their_prefix_in_the_budget():
    text = ' '.join(['слово'] * 200)
    parts = split_message(text, numbered=True)
    assert len(parts) >= 10
    assert all(part.startswith(f'{i}/{len(parts)} ') for i, part in enumerate(parts, 1))
    assert max(sizes(parts)) <= 200
    assert ' '.join(part.split(' ', 1)[1] for part in parts) == text
    assert split_message('short', numbered=True) == ['short']
    assert split_message('word ' * 40, numbered=True) == ['word ' * 40]


def test_send_message_sends_every_part_within_the_limit(iface):
    # 21 bytes a sentence: 9 of them fit in a message
    send_message('привет мир. ' * 100, 42, iface)
    assert len(iface.sent) == 12
    assert {destination for destination, _ in iface.sent} == {42}
    assert ' '.join(text for _, text in iface.sent).rstrip() == ('привет мир. ' * 100).rstrip()


def test_a_send_error_is_logged_and_the_rest_still_goes_out(iface, caplog):
    calls = []

    def failing_send(text, destinationId, wantAck=False, wantResponse=False):
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError('Data payload too big')
        return type('Packet', (), {'id': len(calls)})()

    iface.sendText = failing_send
    with caplog.at_level(logging.INFO):
        send_message('word ' * 100, 42, iface)
    assert len(calls) == 3
    assert 'REPLY SEND ERROR Data payload too big' in caplog.text


def test_sends_from_two_threads_take_turns(iface):
    inside = []
    overlaps = []
    original = iface.sendText

    def slow_send(**kwargs):
        if inside:
            overlaps.append(kwargs['text'])
        inside.append(1)
        time.sleep(0.002)
        inside.pop()
        return original(**kwargs)

    iface.sendText = slow_send
    threads = [threading.Thread(target=send_message, args=(f'thread {n} ' * 150, n, iface)) for n in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not overlaps
    assert len(iface.texts(1)) == len(iface.texts(2)) > 1


def test_a_sync_message_to_another_bbs_goes_whole_up_to_the_radios_limit():
    # A peer parses BULLETIN|... only whole: splitting it at 200 bytes would lose it
    interface = FakeInterface()
    send_bulletin_to_bbs_nodes('General', 'BOB', 'Сбор', 'Собираемся в субботу у фонтана в Ваке.' + 'x' * 79,
                               '0d3c6a1e-1111-2222-3333-444455556666', ['!0c000001'], interface)
    [(destination, text)] = interface.sent
    assert destination == '!0c000001' and 200 < len(text.encode('utf-8')) <= 233
