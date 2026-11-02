"""A message its destination doesn't ACK is sent once more: the radio stops
retrying when it hears a relay pass the message on, which doesn't mean the
node got it."""
import logging

import pytest
from meshtastic import BROADCAST_NUM

import utils
from conftest import BBS_NUM, pass_resend_after
from message_processing import on_receive
from utils import first_sent_as, heard_from, resend_pending, send_bulletin_to_bbs_nodes, send_message

NODE = 0x43569d9c
PEER_BBS = '!0a000002'


@pytest.fixture
def radio(iface):
    iface.add_node(NODE, 'alex')
    iface.add_node(BBS_NUM, 'BBS0')
    return iface


def dm(iface, text, packet_id=1):
    on_receive({'from': NODE, 'fromId': f'!{NODE:08x}', 'to': BBS_NUM, 'id': packet_id,
                'decoded': {'portnum': 'TEXT_MESSAGE_APP', 'payload': text.encode()}}, iface)


def test_a_message_only_a_relay_passed_on_is_sent_again_once(radio, resends, caplog):
    with caplog.at_level(logging.INFO):
        # FakeInterface answers each message with the implicit ACK: a relay passed it on, nothing more
        assert send_message('Test 17', NODE, radio) == [1]
        [(delay, _, _)] = resends
        assert delay == utils.RESEND_AFTER == 45
        assert resend_pending(1)
        pass_resend_after(resends, 1)
    assert radio.texts(NODE) == ['Test 17', 'Test 17']
    assert not resend_pending(1)
    assert first_sent_as(2) == 1
    assert 'sendID 1 got no ACK from !43569d9c in 45 s, sending it again' in caplog.text
    assert "Sending message again (first sendID 1) to user 'alex' (!43569d9c) with sendID 2" in caplog.text
    # Once: the copy has no resend of its own
    assert len(resends) == 1


def test_a_message_the_node_acked_is_not_sent_again(radio, resends):
    send_message('Test 18', NODE, radio)
    radio.ack(1, NODE)
    pass_resend_after(resends, 1)
    assert radio.texts(NODE) == ['Test 18']


@pytest.mark.parametrize('reason, again', [
    ('MAX_RETRANSMIT', True),       # lost on air
    ('PKI_UNKNOWN_PUBKEY', False),  # the same copy would fail the same way
])
def test_a_nak_sends_it_again_only_if_the_air_lost_it(radio, resends, caplog, reason, again):
    radio.acks = False
    with caplog.at_level(logging.INFO):
        send_message('Test 24', NODE, radio)
        radio.answer(reason)
        pass_resend_after(resends, 1)
    assert radio.texts(NODE) == ['Test 24'] * (2 if again else 1)
    if not again:
        assert f'sendID 1 got no ACK from !43569d9c, not sending it again: {reason}' in caplog.text


def test_a_reply_is_not_sent_again_once_the_node_wrote_since(radio, resends, caplog):
    dm(radio, 'Test 18')
    [menu] = radio.texts(NODE)
    # The node got the menu but its ACK was lost, or it never got it and asks again: either way it moved on
    dm(radio, 'Test 19', packet_id=2)
    assert radio.texts(NODE) == [menu, menu]
    with caplog.at_level(logging.INFO):
        pass_resend_after(resends, 1)
    assert radio.texts(NODE) == [menu, menu]
    assert 'sendID 1 got no ACK from !43569d9c, not sending it again: the node wrote since' in caplog.text
    # The answer to the newer message still gets its resend
    pass_resend_after(resends, 2)
    assert radio.texts(NODE) == [menu, menu, menu]


def test_a_message_nobody_asked_for_is_sent_again_although_the_node_wrote(radio, resends):
    send_message('wife: Buy milk', NODE, radio, reply=False)
    heard_from(NODE)
    pass_resend_after(resends, 1)
    assert radio.texts(NODE) == ['wife: Buy milk', 'wife: Buy milk']


def test_broadcasts_and_sync_messages_are_never_sent_again(radio, resends):
    send_message('Urgent bulletin', BROADCAST_NUM, radio)
    # A peer BBS would store a bulletin it got twice twice
    send_bulletin_to_bbs_nodes('General', 'alex', 'Hi', 'Hello', 'uuid-1', [PEER_BBS], radio)
    assert len(radio.sent) == 2
    assert resends == []


def test_a_destination_given_by_its_node_id_counts_its_ack(radio, resends):
    send_message('Hi', f'!{NODE:08x}', radio)
    radio.ack(1, NODE)
    pass_resend_after(resends, 1)
    assert len(radio.sent) == 1


def test_the_nodes_ack_after_the_implicit_ack_is_logged_once(radio, caplog):
    with caplog.at_level(logging.INFO):
        send_message('Test 31', NODE, radio)
        radio.ack(1, NODE)
        # Its ACK of the radio's retransmission
        radio.ack(1, NODE)
    assert 'sendID 1 was relayed (implicit ACK)' in caplog.text
    assert caplog.text.count(f'sendID 1 was ACKed by {NODE}') == 1


def test_the_nodes_ack_as_the_first_answer_is_logged_once(radio, caplog):
    radio.acks = False
    with caplog.at_level(logging.INFO):
        send_message('Test 41', NODE, radio)
        # As meshtastic does it: the response handler first, then the packet to the subscribers
        packet_id, on_response = radio.unanswered.pop(0)
        ack = radio.routing(packet_id, NODE)
        on_response(ack)
        radio.publish(ack)
    assert caplog.text.count(f'sendID 1 was ACKed by {NODE}') == 1


def test_a_routing_answer_it_cannot_read_never_raises_on_the_publishing_thread(radio, caplog):
    with caplog.at_level(logging.INFO):
        utils.on_packet({'decoded': {'portnum': 'ROUTING_APP', 'requestId': 1, 'routing': 'garbage'}}, radio)
    assert 'Routing answer not understood' in caplog.text
