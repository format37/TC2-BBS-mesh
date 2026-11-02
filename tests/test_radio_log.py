"""What the logs say about the radio path: the radio's own log in a file,
the route of every received message, and who answered every sent one."""
import logging

import pytest
from pubsub import pub

import radio_log
from conftest import BBS_NUM
from message_processing import on_receive
from utils import describe_route, send_message


@pytest.fixture
def radio_file(tmp_path):
    path = tmp_path / 'radio.log'
    handler = radio_log.start(str(path))
    yield path
    radio_log.stop(handler)


class Collect(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def test_the_radio_log_lines_go_to_their_own_file_without_colours(radio_file):
    line = ('\x1b[0m\x1b[34mDEBUG \x1b[0m| 12:52:37 208 [Router] \x1b[34m'
            'Setting next hop for packet with dest 43569d9c to 9c\n')
    bbs_log = Collect()
    logging.getLogger().addHandler(bbs_log)
    try:
        pub.sendMessage(radio_log.TOPIC, line=line, interface=None)
    finally:
        logging.getLogger().removeHandler(bbs_log)
    text = radio_file.read_text(encoding='utf-8')
    assert text.endswith(' DEBUG | 12:52:37 208 [Router] Setting next hop for packet with dest 43569d9c to 9c\n')
    # Not in the BBS log, which it would drown
    assert bbs_log.lines == []


def test_the_radio_log_stops_with_its_handler(tmp_path):
    path = tmp_path / 'radio.log'
    radio_log.stop(radio_log.start(str(path)))
    pub.sendMessage(radio_log.TOPIC, line='after', interface=None)
    assert 'after' not in path.read_text(encoding='utf-8')


@pytest.mark.parametrize('packet, route', [
    ({'hopStart': 7, 'hopLimit': 7, 'relayNode': 0x9c, 'rxSnr': 6.25, 'rxRssi': -20},
     'direct, relay 0x9c, SNR 6.25, RSSI -20'),
    ({'hopStart': 7, 'hopLimit': 6, 'relayNode': 0x2c, 'rxSnr': -12.5, 'rxRssi': -105},
     '1 hop, relay 0x2c, SNR -12.5, RSSI -105'),
    # meshtastic leaves out a hopLimit of 0
    ({'hopStart': 3, 'relayNode': 0x03, 'rxSnr': -7.0}, '3 hops, relay 0x03, SNR -7.0'),
    ({}, 'route unknown'),
])
def test_the_route_of_a_packet(packet, route):
    assert describe_route(packet) == route


def test_a_received_message_is_logged_with_its_route(iface, caplog):
    iface.add_node(0x43569d9c, 'alex')
    iface.add_node(BBS_NUM, 'BBS0')
    packet = {'from': 0x43569d9c, 'fromId': '!43569d9c', 'to': BBS_NUM, 'id': 1,
              'hopStart': 7, 'hopLimit': 6, 'relayNode': 0x2c, 'rxSnr': -12.5, 'rxRssi': -105,
              'decoded': {'portnum': 'TEXT_MESSAGE_APP', 'payload': 'Test 4'.encode()}}
    with caplog.at_level(logging.INFO):
        on_receive(packet, iface)
    assert ("Received message from user 'alex' (!43569d9c) to BBS0 "
            "(1 hop, relay 0x2c, SNR -12.5, RSSI -105): Test 4") in caplog.text
    assert iface.texts(0x43569d9c), 'the BBS still answers'


def answers(iface, caplog, respond):
    iface.acks = False
    with caplog.at_level(logging.INFO):
        send_message('hi', 0x43569d9c, iface)
        packet_id, on_response = iface.unanswered.pop(0)
        on_response(respond(packet_id))
    return caplog.text


def test_who_answered_a_sent_message_is_logged(iface, caplog):
    def ack(packet_id):
        return {'from': 0x43569d9c, 'fromId': '!43569d9c', 'hopStart': 2, 'hopLimit': 1, 'relayNode': 0x2c,
                'rxSnr': -10.0, 'rxRssi': -101,
                'decoded': {'portnum': 'ROUTING_APP', 'requestId': packet_id, 'routing': {}}}
    assert 'sendID 1 was ACKed by !43569d9c (1 hop, relay 0x2c, SNR -10.0, RSSI -101)' in answers(iface, caplog, ack)


def test_a_relay_heard_by_the_radio_is_logged_as_an_implicit_ack(iface, caplog):
    assert 'sendID 1 was relayed (implicit ACK)' in answers(iface, caplog, lambda i: iface.routing(i, BBS_NUM))


def test_a_nak_is_logged_as_not_delivered(iface, caplog):
    text = answers(iface, caplog, lambda i: iface.routing(i, BBS_NUM, 'MAX_RETRANSMIT'))
    assert 'sendID 1 was not delivered: MAX_RETRANSMIT' in text


def test_an_answer_it_cannot_read_never_raises_on_the_reader_thread(iface, caplog):
    assert 'sendID answer not understood' in answers(iface, caplog, lambda i: {'decoded': 'garbage'})
