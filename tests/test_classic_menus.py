"""Fixes in the classic menus."""
from db_operations import add_bulletin
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

