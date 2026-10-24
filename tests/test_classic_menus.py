"""Fixes in the classic menus and BBS sync."""
from conftest import FakeInterface
from db_operations import add_bulletin, delete_bulletin, get_bulletin, get_bulletin_id
from message_processing import process_message


def test_cb_then_a_number_reads_the_bulletin(iface):
    alice = iface.add_node(0x11111111, 'ALC')
    add_bulletin('General', 'ALC', 'Party', 'Saturday at 8', [], iface, sender_id='!11111111')
    process_message(alice, 'CB,,general', iface)
    process_message(alice, '1', iface)
    assert iface.texts(alice)[-1].endswith('Subject: Party\n\nSaturday at 8')


def test_a_long_russian_bulletin_reaches_the_radio_whole(iface):
    alice = iface.add_node(0x11111111, 'ALC')
    content = 'Собираемся в субботу у фонтана, берите фонарики. ' * 12
    add_bulletin('General', 'ALC', 'Сбор', content, [], iface, sender_id='!11111111')
    process_message(alice, 'CB,,general', iface)
    process_message(alice, '1', iface)
    # Every part fit (the iface fixture checks), and nothing was lost
    received = ' '.join(iface.texts(alice)[1:])
    assert received.split()[-24:] == content.split()[-24:]
    assert len(iface.texts(alice)) > 3


def test_a_deleted_bulletin_is_deleted_on_peers_by_its_unique_id():
    interface = FakeInterface()
    party = get_bulletin_id(add_bulletin('General', 'BOB', 'Party', 'x', [], None, unique_id='u-party'))
    rules = get_bulletin_id(add_bulletin('Info', 'BOB', 'Rules', 'x', [], None, unique_id='u-rules'))
    delete_bulletin(rules, ['!0c000001'], interface)
    assert interface.sent == [('!0c000001', 'DELETE_BULLETIN|u-rules')]

    # A peer numbers bulletins its own way: the unique_id finds the right one
    peer = interface.add_node(0x0c000001, 'BBS2')
    interface.bbs_nodes = ['!0c000001']
    other = get_bulletin_id(add_bulletin('News', 'REM', 'Other', 'x', [], None, unique_id='u-other'))
    process_message(peer, 'DELETE_BULLETIN|u-party', interface, is_sync_message=True)
    assert get_bulletin(party) is None and get_bulletin(other) is not None
    # An old peer sends its local id: nothing is deleted, rather than a wrong bulletin
    process_message(peer, f'DELETE_BULLETIN|{other}', interface, is_sync_message=True)
    assert get_bulletin(other) is not None
