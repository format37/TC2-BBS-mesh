import configparser
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor

from meshtastic import BROADCAST_NUM

from ai_tools import answer_of, text_for

from db_operations import (
    add_bulletin, add_mail, delete_mail, delete_bulletin,
    get_bulletin_content, get_bulletins,
    get_mail, get_mail_content,
    add_channel, get_channels, delete_channel, get_sender_id_by_mail_id,
    add_conversation_message, get_conversation_history, clear_conversation_history,
    get_last_users,
    get_recent_telegram_contact, get_telegram_contact, get_telegram_contacts,
    get_telegram_share_position, purge_expired_telegram_invites, set_telegram_share_position,
    touch_telegram_contact
)
from telegram_bridge import RADIO_NOT_SENT, contact_name_problem
from utils import (
    change_user_state, get_node_id_from_num, get_node_info,
    get_node_short_name, get_user_state, send_message,
    update_user_state
)

# Read the configuration for menu options
config = configparser.ConfigParser()
config.read('config.ini')

main_menu_items = config['menu']['main_menu_items'].split(',')
bbs_menu_items = config['menu']['bbs_menu_items'].split(',')
utilities_menu_items = config['menu']['utilities_menu_items'].split(',')

# Get admin node for bulletin deletion privileges
admin_node = config.get('admin', 'admin_node', fallback='').strip()


def is_admin_or_owner(sender_id, bulletin_sender_id, interface):
    """Check if sender can delete a bulletin (is admin or owner)."""
    node_id = get_node_id_from_num(sender_id, interface)
    # Admin can delete any bulletin
    if admin_node and node_id == admin_node:
        return True
    # Owner can delete their own bulletin
    if bulletin_sender_id and node_id == bulletin_sender_id:
        return True
    return False


def build_menu(items, menu_name):
    menu_str = f"{menu_name}\n"
    for item in items:
        if item.strip() == 'Q':
            menu_str += "[Q]uick Commands\n"
        elif item.strip() == 'B':
            if menu_name == "📰BBS Menu📰":
                menu_str += "[B]ulletins\n"
            else:
                menu_str += "[B]BS\n"
        elif item.strip() == 'U':
            menu_str += "[U]tilities\n"
        elif item.strip() == 'X':
            menu_str += "E[X]IT\n"
        elif item.strip() == 'M':
            menu_str += "[M]ail\n"
        elif item.strip() == 'C':
            menu_str += "[C]hannel Dir\n"
        elif item.strip() == 'J':
            menu_str += "[J]S8CALL\n"
        elif item.strip() == 'S':
            menu_str += "[S]tats\n"
        elif item.strip() == 'F':
            menu_str += "[F]ortune\n"
        elif item.strip() == 'W':
            menu_str += "[W]all of Shame\n"
        elif item.strip() == 'A':
            menu_str += "[A]I Chat\n"
        elif item.strip() == 'T':
            menu_str += "[T]elegram\n"
        elif item.strip() == 'L':
            menu_str += "[L]ast users\n"
    return menu_str

def handle_help_command(sender_id, interface, menu_name=None):
    if menu_name:
        update_user_state(sender_id, {'command': 'MENU', 'menu': menu_name, 'step': 1})
        if menu_name == 'bbs':
            response = build_menu(bbs_menu_items, "📰BBS Menu📰")
        elif menu_name == 'utilities':
            response = build_menu(utilities_menu_items, "🛠️Utilities Menu🛠️")
    else:
        update_user_state(sender_id, {'command': 'MAIN_MENU', 'step': 1})  # Reset to main menu state
        mail = get_mail(get_node_id_from_num(sender_id, interface))
        response = build_menu(main_menu_items, f"💾TC² BBS💾 (✉️:{len(mail)})")
    send_message(response, sender_id, interface)

def get_node_name(node_id, interface):
    node_info = interface.nodes.get(node_id)
    if node_info:
        return node_info['user']['longName']
    return f"Node {node_id}"


def handle_mail_command(sender_id, interface):
    response = "✉️Mail Menu✉️\nWhat would you like to do with mail?\n[R]ead  [S]end E[X]IT"
    send_message(response, sender_id, interface)
    update_user_state(sender_id, {'command': 'MAIL', 'step': 1})



def handle_bulletin_command(sender_id, interface):
    response = f"📰Bulletin Menu📰\nWhich board would you like to enter?\n[G]eneral  [I]nfo  [N]ews  [U]rgent"
    send_message(response, sender_id, interface)
    update_user_state(sender_id, {'command': 'BULLETIN_MENU', 'step': 1})


def handle_exit_command(sender_id, interface):
    send_message("Type 'HELP' for a list of commands.", sender_id, interface)
    update_user_state(sender_id, None)


def handle_stats_command(sender_id, interface):
    response = "📊Stats Menu📊\nWhat stats would you like to view?\n[N]odes  [H]ardware  [R]oles  E[X]IT"
    send_message(response, sender_id, interface)
    update_user_state(sender_id, {'command': 'STATS', 'step': 1})


def handle_fortune_command(sender_id, interface):
    try:
        with open('fortunes.txt', 'r') as file:
            fortunes = file.readlines()
        if not fortunes:
            send_message("No fortunes available.", sender_id, interface)
            return
        fortune = random.choice(fortunes).strip()
        decorated_fortune = f"🔮 {fortune} 🔮"
        send_message(decorated_fortune, sender_id, interface)
    except Exception as e:
        send_message(f"Error generating fortune: {e}", sender_id, interface)


def handle_stats_steps(sender_id, message, step, interface):
    message = message.lower().strip()
    if len(message) == 2 and message[1] == 'x':
        message = message[0]

    if step == 1:
        choice = message
        if choice == 'x':
            handle_help_command(sender_id, interface)
            return
        elif choice == 'n':
            current_time = int(time.time())
            timeframes = {
                "All time": None,
                "Last 24 hours": 86400,
                "Last 8 hours": 28800,
                "Last hour": 3600
            }
            total_nodes_summary = []

            for period, seconds in timeframes.items():
                if seconds is None:
                    total_nodes = len(interface.nodes)
                else:
                    time_limit = current_time - seconds
                    total_nodes = sum(1 for node in interface.nodes.values() if node.get('lastHeard') is not None and node['lastHeard'] >= time_limit)
                total_nodes_summary.append(f"- {period}: {total_nodes}")

            response = "Total nodes seen:\n" + "\n".join(total_nodes_summary)
            send_message(response, sender_id, interface)
            handle_stats_command(sender_id, interface)
        elif choice == 'h':
            hw_models = {}
            for node in interface.nodes.values():
                hw_model = node['user'].get('hwModel', 'Unknown')
                hw_models[hw_model] = hw_models.get(hw_model, 0) + 1
            response = "Hardware Models:\n" + "\n".join([f"{model}: {count}" for model, count in hw_models.items()])
            send_message(response, sender_id, interface)
            handle_stats_command(sender_id, interface)
        elif choice == 'r':
            roles = {}
            for node in interface.nodes.values():
                role = node['user'].get('role', 'Unknown')
                roles[role] = roles.get(role, 0) + 1
            response = "Roles:\n" + "\n".join([f"{role}: {count}" for role, count in roles.items()])
            send_message(response, sender_id, interface)
            handle_stats_command(sender_id, interface)


def handle_bb_steps(sender_id, message, step, state, interface, bbs_nodes):
    boards = {0: "General", 1: "Info", 2: "News", 3: "Urgent"}
    if step == 1:
        if message.lower() == 'e':
            handle_help_command(sender_id, interface, 'bbs')
            return
        board_name = boards[int(message)]
        bulletins = get_bulletins(board_name)
        response = f"{board_name} has {len(bulletins)} messages.\n[R]ead  [P]ost  [D]elete"
        send_message(response, sender_id, interface)
        update_user_state(sender_id, {'command': 'BULLETIN_ACTION', 'step': 2, 'board': board_name})

    elif step == 2:
        board_name = state['board']
        if message.lower() == 'r':
            bulletins = get_bulletins(board_name)
            if bulletins:
                send_message(f"Select a bulletin number to view from {board_name}:", sender_id, interface)
                for bulletin in bulletins:
                    send_message(f"[{bulletin[0]}] {bulletin[1]}", sender_id, interface)
                update_user_state(sender_id, {'command': 'BULLETIN_READ', 'step': 3, 'board': board_name})
            else:
                send_message(f"No bulletins in {board_name}.", sender_id, interface)
                handle_bb_steps(sender_id, 'e', 1, state, interface, bbs_nodes)
        elif message.lower() == 'p':
            if board_name.lower() == 'urgent':
                node_id = get_node_id_from_num(sender_id, interface)
                allowed_nodes = interface.allowed_nodes
                logging.info(f"Checking permissions for node_id: {node_id} with allowed_nodes: {allowed_nodes}")  # Debug statement
                if allowed_nodes and node_id not in allowed_nodes:
                    send_message("You don't have permission to post to this board.", sender_id, interface)
                    handle_bb_steps(sender_id, 'e', 1, state, interface, bbs_nodes)
                    return
            send_message("What is the subject of your bulletin? Keep it short.", sender_id, interface)
            update_user_state(sender_id, {'command': 'BULLETIN_POST', 'step': 4, 'board': board_name})
        elif message.lower() == 'd':
            bulletins = get_bulletins(board_name)
            if bulletins:
                send_message(f"Select bulletin to delete from {board_name}:", sender_id, interface)
                for bulletin in bulletins:
                    send_message(f"[{bulletin[0]}] {bulletin[1]}", sender_id, interface)
                update_user_state(sender_id, {'command': 'BULLETIN_DELETE', 'step': 6, 'board': board_name})
            else:
                send_message(f"No bulletins in {board_name}.", sender_id, interface)
                handle_bb_steps(sender_id, 'e', 1, state, interface, bbs_nodes)

    elif step == 3:
        bulletin_id = int(message)
        sender_short_name, date, subject, content, unique_id, bulletin_sender_id = get_bulletin_content(bulletin_id)
        send_message(f"From: {sender_short_name}\nDate: {date}\nSubject: {subject}\n- - - - - - -\n{content}", sender_id, interface)
        board_name = state['board']
        handle_bb_steps(sender_id, 'e', 1, state, interface, bbs_nodes)

    elif step == 4:
        subject = message
        send_message("Send the contents of your bulletin. Send a message with END when finished.", sender_id, interface)
        update_user_state(sender_id, {'command': 'BULLETIN_POST_CONTENT', 'step': 5, 'board': state['board'], 'subject': subject, 'content': ''})

    elif step == 5:
        if message.lower() == "end":
            board = state['board']
            subject = state['subject']
            content = state['content']
            node_id = get_node_id_from_num(sender_id, interface)
            node_info = interface.nodes.get(node_id)
            if node_info is None:
                send_message("Error: Unable to retrieve your node information.", sender_id, interface)
                update_user_state(sender_id, None)
                return
            sender_short_name = node_info['user'].get('shortName', f"Node {sender_id}")
            unique_id = add_bulletin(board, sender_short_name, subject, content, bbs_nodes, interface, sender_id=node_id)
            send_message(f"Your bulletin '{subject}' has been posted to {board}.\n(╯°□°)╯📄📌[{board}]", sender_id, interface)
            handle_bb_steps(sender_id, 'e', 1, state, interface, bbs_nodes)
        else:
            state['content'] += message + "\n"
            update_user_state(sender_id, state)

    elif step == 6:
        # Delete bulletin step
        try:
            bulletin_id = int(message)
            bulletin_data = get_bulletin_content(bulletin_id)
            if bulletin_data is None:
                send_message("Bulletin not found.", sender_id, interface)
                handle_bb_steps(sender_id, 'e', 1, state, interface, bbs_nodes)
                return

            sender_short_name, date, subject, content, unique_id, bulletin_sender_id = bulletin_data

            # Check permission: admin or owner
            if is_admin_or_owner(sender_id, bulletin_sender_id, interface):
                delete_bulletin(bulletin_id, bbs_nodes if bbs_nodes else [], interface)
                send_message(f"Bulletin '{subject}' deleted.", sender_id, interface)
            else:
                send_message("Permission denied. You can only delete your own bulletins.", sender_id, interface)

            handle_bb_steps(sender_id, 'e', 1, state, interface, bbs_nodes)
        except ValueError:
            send_message("Invalid bulletin number.", sender_id, interface)
            handle_bb_steps(sender_id, 'e', 1, state, interface, bbs_nodes)


def handle_mail_steps(sender_id, message, step, state, interface, bbs_nodes):
    message = message.strip()
    if len(message) == 2 and message[1] == 'x':
        message = message[0]

    if step == 1:
        choice = message.lower()
        if choice == 'r':
            sender_node_id = get_node_id_from_num(sender_id, interface)
            mail = get_mail(sender_node_id)
            if mail:
                send_message(f"You have {len(mail)} mail messages. Select a message number to read:", sender_id, interface)
                for msg in mail:
                    send_message(f"-{msg[0]}-\nDate: {msg[3]}\nFrom: {msg[1]}\nSubject: {msg[2]}", sender_id, interface)
                update_user_state(sender_id, {'command': 'MAIL', 'step': 2})
            else:
                send_message("There are no messages in your mailbox.📭", sender_id, interface)
                update_user_state(sender_id, None)
        elif choice == 's':
            send_message("What is the Short Name of the node you want to leave a message for?", sender_id, interface)
            update_user_state(sender_id, {'command': 'MAIL', 'step': 3})
        elif choice == 'x':
            handle_help_command(sender_id, interface)

    elif step == 2:
        mail_id = int(message)
        try:
            sender_node_id = get_node_id_from_num(sender_id, interface)
            sender, date, subject, content, unique_id = get_mail_content(mail_id, sender_node_id)
            send_message(f"Date: {date}\nFrom: {sender}\nSubject: {subject}\n{content}", sender_id, interface)
            send_message("What would you like to do with this message?\n[K]eep  [D]elete  [R]eply", sender_id, interface)
            update_user_state(sender_id, {'command': 'MAIL', 'step': 4, 'mail_id': mail_id, 'unique_id': unique_id, 'sender': sender, 'subject': subject, 'content': content})
        except TypeError:
            logging.info(f"Node {sender_id} tried to access non-existent message")
            send_message("Mail not found", sender_id, interface)
            update_user_state(sender_id, None)

    elif step == 3:
        short_name = message.lower()
        nodes = get_node_info(interface, short_name)
        if not nodes:
            send_message("I'm unable to find that node in my database.", sender_id, interface)
            handle_mail_command(sender_id, interface)
        elif len(nodes) == 1:
            recipient_id = nodes[0]['num']
            recipient_name = get_node_name(recipient_id, interface)
            send_message(f"What is the subject of your message to {recipient_name}?\nKeep it short.", sender_id, interface)
            update_user_state(sender_id, {'command': 'MAIL', 'step': 5, 'recipient_id': recipient_id})
        else:
            send_message("There are multiple nodes with that short name. Which one would you like to leave a message for?", sender_id, interface)
            for i, node in enumerate(nodes):
                send_message(f"[{i}] {node['longName']}", sender_id, interface)
            update_user_state(sender_id, {'command': 'MAIL', 'step': 6, 'nodes': nodes})

    elif step == 4:
        if message.lower() == "d":
            unique_id = state['unique_id']
            sender_node_id = get_node_id_from_num(sender_id, interface)
            delete_mail(unique_id, sender_node_id, bbs_nodes, interface)
            send_message("The message has been deleted 🗑️", sender_id, interface)
            update_user_state(sender_id, None)
        elif message.lower() == "r":
            sender = state['sender']
            send_message(f"Send your reply to {sender} now, followed by a message with END", sender_id, interface)
            update_user_state(sender_id, {'command': 'MAIL', 'step': 7, 'reply_to_mail_id': state['mail_id'], 'subject': f"Re: {state['subject']}", 'content': ''})
        else:
            send_message("The message has been kept in your inbox.✉️", sender_id, interface)
            update_user_state(sender_id, None)

    elif step == 5:
        subject = message
        send_message("Send your message. You can send it in multiple messages if it's too long for one.\nSend a single message with END when you're done", sender_id, interface)
        update_user_state(sender_id, {'command': 'MAIL', 'step': 7, 'recipient_id': state['recipient_id'], 'subject': subject, 'content': ''})

    elif step == 6:
        selected_node_index = int(message)
        selected_node = state['nodes'][selected_node_index]
        recipient_id = selected_node['num']
        recipient_name = get_node_name(recipient_id, interface)
        send_message(f"What is the subject of your message to {recipient_name}?\nKeep it short.", sender_id, interface)
        update_user_state(sender_id, {'command': 'MAIL', 'step': 5, 'recipient_id': recipient_id})

    elif step == 7:
        if message.lower() == "end":
            if 'reply_to_mail_id' in state:
                recipient_id = get_sender_id_by_mail_id(state['reply_to_mail_id'])  # Get the sender ID from the mail ID
            else:
                recipient_id = state.get('recipient_id')
            subject = state['subject']
            content = state['content']
            recipient_name = get_node_name(recipient_id, interface)

            sender_short_name = get_node_short_name(get_node_id_from_num(sender_id, interface), interface)
            unique_id = add_mail(get_node_id_from_num(sender_id, interface), sender_short_name, recipient_id, subject, content, bbs_nodes, interface)
            send_message(f"Mail has been posted to the mailbox of {recipient_name}.\n(╯°□°)╯📨📬", sender_id, interface)

            notification_message = f"You have a new mail message from {sender_short_name}. Check your mailbox by responding to this message with CM."
            send_message(notification_message, recipient_id, interface)

            update_user_state(sender_id, None)
            update_user_state(sender_id, {'command': 'MAIL', 'step': 8})
        else:
            state['content'] += message + "\n"
            update_user_state(sender_id, state)

    elif step == 8:
        if message.lower() == "y":
            handle_mail_command(sender_id, interface)
        else:
            send_message("Okay, feel free to send another command.", sender_id, interface)
            update_user_state(sender_id, None)


def handle_wall_of_shame_command(sender_id, interface):
    response = "Devices with battery levels below 20%:\n"
    for node_id, node in interface.nodes.items():
        metrics = node.get('deviceMetrics', {})
        battery_level = metrics.get('batteryLevel', 101)
        if battery_level < 20:
            long_name = node['user']['longName']
            response += f"{long_name} - Battery {battery_level}%\n"
    if response == "Devices with battery levels below 20%:\n":
        response = "No devices with battery levels below 20% found."
    send_message(response, sender_id, interface)


def handle_channel_directory_command(sender_id, interface):
    response = "📚CHANNEL DIRECTORY📚\nWhat would you like to do?\n[V]iew [P]ost [D]elete E[X]IT"
    send_message(response, sender_id, interface)
    update_user_state(sender_id, {'command': 'CHANNEL_DIRECTORY', 'step': 1})


def handle_channel_directory_steps(sender_id, message, step, state, interface):
    message = message.strip()
    if len(message) == 2 and message[1] == 'x':
        message = message[0]

    if step == 1:
        choice = message
        if choice.lower() == 'x':
            handle_help_command(sender_id, interface)
            return
        elif choice.lower() == 'v':
            channels = get_channels()
            if channels:
                response = "Select a channel number to view:\n" + "\n".join(
                    [f"[{i}] {channel[1]}" for i, channel in enumerate(channels)])
                send_message(response, sender_id, interface)
                update_user_state(sender_id, {'command': 'CHANNEL_DIRECTORY', 'step': 2})
            else:
                send_message("No channels available in the directory.", sender_id, interface)
                handle_channel_directory_command(sender_id, interface)
        elif choice.lower() == 'p':
            send_message("Name your channel for the directory:", sender_id, interface)
            update_user_state(sender_id, {'command': 'CHANNEL_DIRECTORY', 'step': 3})
        elif choice.lower() == 'd':
            # Check if admin
            node_id = get_node_id_from_num(sender_id, interface)
            if not admin_node or node_id != admin_node:
                send_message("Permission denied. Admin only.", sender_id, interface)
                handle_channel_directory_command(sender_id, interface)
                return
            channels = get_channels()
            if channels:
                response = "Select channel to delete:\n" + "\n".join(
                    [f"[{i}] {channel[1]}" for i, channel in enumerate(channels)])
                send_message(response, sender_id, interface)
                update_user_state(sender_id, {'command': 'CHANNEL_DIRECTORY', 'step': 5})
            else:
                send_message("No channels to delete.", sender_id, interface)
                handle_channel_directory_command(sender_id, interface)

    elif step == 2:
        channel_index = int(message)
        channels = get_channels()
        if 0 <= channel_index < len(channels):
            channel_id, channel_name, channel_url = channels[channel_index]
            send_message(f"Channel Name: {channel_name}\nChannel URL:\n{channel_url}", sender_id, interface)
        handle_channel_directory_command(sender_id, interface)

    elif step == 3:
        channel_name = message
        send_message("Send a message with your channel URL or PSK:", sender_id, interface)
        update_user_state(sender_id, {'command': 'CHANNEL_DIRECTORY', 'step': 4, 'channel_name': channel_name})

    elif step == 4:
        channel_url = message
        channel_name = state['channel_name']
        add_channel(channel_name, channel_url)
        send_message(f"Your channel '{channel_name}' has been added to the directory.", sender_id, interface)
        handle_channel_directory_command(sender_id, interface)

    elif step == 5:
        # Delete channel (admin only)
        try:
            channel_index = int(message)
            channels = get_channels()
            if 0 <= channel_index < len(channels):
                channel_id, channel_name, channel_url = channels[channel_index]
                delete_channel(channel_id)
                send_message(f"Channel '{channel_name}' deleted.", sender_id, interface)
            else:
                send_message("Invalid channel number.", sender_id, interface)
        except ValueError:
            send_message("Invalid channel number.", sender_id, interface)
        handle_channel_directory_command(sender_id, interface)


def handle_send_mail_command(sender_id, message, interface, bbs_nodes):
    try:
        parts = message.split(",,", 3)
        if len(parts) != 4:
            send_message("Send Mail Quick Command format:\nSM,,{short_name},,{subject},,{message}", sender_id, interface)
            return

        _, short_name, subject, content = parts
        nodes = get_node_info(interface, short_name.lower())
        if not nodes:
            send_message(f"Node with short name '{short_name}' not found.", sender_id, interface)
            return
        if len(nodes) > 1:
            send_message(f"Multiple nodes with short name '{short_name}' found. Please be more specific.", sender_id,
                         interface)
            return

        recipient_id = nodes[0]['num']
        recipient_name = get_node_name(recipient_id, interface)
        sender_short_name = get_node_short_name(get_node_id_from_num(sender_id, interface), interface)

        unique_id = add_mail(get_node_id_from_num(sender_id, interface), sender_short_name, recipient_id, subject,
                             content, bbs_nodes, interface)
        send_message(f"Mail has been sent to {recipient_name}.", sender_id, interface)

        notification_message = f"You have a new mail message from {sender_short_name}. Check your mailbox by responding to this message with CM."
        send_message(notification_message, recipient_id, interface)

    except Exception as e:
        logging.error(f"Error processing send mail command: {e}")
        send_message("Error processing send mail command.", sender_id, interface)


def handle_check_mail_command(sender_id, interface):
    try:
        sender_node_id = get_node_id_from_num(sender_id, interface)
        mail = get_mail(sender_node_id)
        if not mail:
            send_message("You have no new messages.", sender_id, interface)
            return

        response = "📬 You have the following messages:\n"
        for i, msg in enumerate(mail):
            response += f"{i + 1:02d}. From: {msg[1]}, Subject: {msg[2]}\n"
        response += "\nPlease reply with the number of the message you want to read."
        send_message(response, sender_id, interface)

        update_user_state(sender_id, {'command': 'CHECK_MAIL', 'step': 1, 'mail': mail})

    except Exception as e:
        logging.error(f"Error processing check mail command: {e}")
        send_message("Error processing check mail command.", sender_id, interface)


def handle_read_mail_command(sender_id, message, state, interface):
    try:
        mail = state.get('mail', [])
        message_number = int(message) - 1

        if message_number < 0 or message_number >= len(mail):
            send_message("Invalid message number. Please try again.", sender_id, interface)
            return

        mail_id = mail[message_number][0]
        sender_node_id = get_node_id_from_num(sender_id, interface)
        sender, date, subject, content, unique_id = get_mail_content(mail_id, sender_node_id)
        response = f"Date: {date}\nFrom: {sender}\nSubject: {subject}\n\n{content}"
        send_message(response, sender_id, interface)
        send_message("What would you like to do with this message?\n[K]eep  [D]elete  [R]eply", sender_id, interface)
        update_user_state(sender_id, {'command': 'CHECK_MAIL', 'step': 2, 'mail_id': mail_id, 'unique_id': unique_id, 'sender': sender, 'subject': subject, 'content': content})

    except ValueError:
        send_message("Invalid input. Please enter a valid message number.", sender_id, interface)
    except Exception as e:
        logging.error(f"Error processing read mail command: {e}")
        send_message("Error processing read mail command.", sender_id, interface)


def handle_delete_mail_confirmation(sender_id, message, state, interface, bbs_nodes):
    try:
        choice = message.lower().strip()
        if len(choice) == 2 and choice[1] == 'x':
            choice = choice[0]

        if choice == 'd':
            unique_id = state['unique_id']
            sender_node_id = get_node_id_from_num(sender_id, interface)
            delete_mail(unique_id, sender_node_id, bbs_nodes, interface)
            send_message("The message has been deleted 🗑️", sender_id, interface)
            update_user_state(sender_id, None)
        elif choice == 'r':
            sender = state['sender']
            send_message(f"Send your reply to {sender} now, followed by a message with END", sender_id, interface)
            update_user_state(sender_id, {'command': 'MAIL', 'step': 7, 'reply_to_mail_id': state['mail_id'], 'subject': f"Re: {state['subject']}", 'content': ''})
        else:
            send_message("The message has been kept in your inbox.✉️", sender_id, interface)
            update_user_state(sender_id, None)

    except Exception as e:
        logging.error(f"Error processing delete mail confirmation: {e}")
        send_message("Error processing delete mail confirmation.", sender_id, interface)



def handle_post_bulletin_command(sender_id, message, interface, bbs_nodes):
    try:
        parts = message.split(",,", 3)
        if len(parts) != 4:
            send_message("Post Bulletin Quick Command format:\nPB,,{board_name},,{subject},,{content}", sender_id, interface)
            return

        _, board_name, subject, content = parts
        node_id = get_node_id_from_num(sender_id, interface)
        if board_name.strip().lower() == 'urgent':
            allowed_nodes = interface.allowed_nodes
            if allowed_nodes and node_id not in allowed_nodes:
                send_message("You don't have permission to post to this board.", sender_id, interface)
                return
        sender_short_name = get_node_short_name(node_id, interface)

        unique_id = add_bulletin(board_name, sender_short_name, subject, content, bbs_nodes, interface, sender_id=node_id)
        send_message(f"Your bulletin '{subject}' has been posted to {board_name}.", sender_id, interface)


    except Exception as e:
        logging.error(f"Error processing post bulletin command: {e}")
        send_message("Error processing post bulletin command.", sender_id, interface)


def handle_check_bulletin_command(sender_id, message, interface):
    try:
        # Split the message only once
        parts = message.split(",,", 1)
        if len(parts) != 2 or not parts[1].strip():
            send_message("Check Bulletins Quick Command format:\nCB,,board_name", sender_id, interface)
            return

        boards = {0: "General", 1: "Info", 2: "News", 3: "Urgent"} #list of boards
        board_name = parts[1].strip().capitalize() #get board name from quick command and capitalize it
        board_name = boards[next(key for key, value in boards.items() if value == board_name)] #search for board name in list

        bulletins = get_bulletins(board_name)
        if not bulletins:
            send_message(f"No bulletins available on {board_name} board.", sender_id, interface)
            return

        response = f"📰 Bulletins on {board_name} board:\n"
        for i, bulletin in enumerate(bulletins):
            response += f"[{i+1:02d}] Subject: {bulletin[1]}, From: {bulletin[2]}, Date: {bulletin[3]}\n"
        response += "\nPlease reply with the number of the bulletin you want to read."
        send_message(response, sender_id, interface)

        update_user_state(sender_id, {'command': 'CHECK_BULLETIN', 'step': 1, 'board_name': board_name, 'bulletins': bulletins})

    except Exception as e:
        logging.error(f"Error processing check bulletin command: {e}")
        send_message("Error processing check bulletin command.", sender_id, interface)

def handle_read_bulletin_command(sender_id, message, state, interface):
    try:
        bulletins = state.get('bulletins', [])
        message_number = int(message) - 1

        if message_number < 0 or message_number >= len(bulletins):
            send_message("Invalid bulletin number. Please try again.", sender_id, interface)
            return

        bulletin_id = bulletins[message_number][0]
        sender, date, subject, content, unique_id, bulletin_sender_id = get_bulletin_content(bulletin_id)
        response = f"Date: {date}\nFrom: {sender}\nSubject: {subject}\n\n{content}"
        send_message(response, sender_id, interface)

        update_user_state(sender_id, None)

    except ValueError:
        send_message("Invalid input. Please enter a valid bulletin number.", sender_id, interface)
    except Exception as e:
        logging.error(f"Error processing read bulletin command: {e}")
        send_message("Error processing read bulletin command.", sender_id, interface)


def handle_post_channel_command(sender_id, message, interface):
    try:
        parts = message.split("|", 3)
        if len(parts) != 3:
            send_message("Post Channel Quick Command format:\nCHP,,{channel_name},,{channel_url}", sender_id, interface)
            return

        _, channel_name, channel_url = parts
        bbs_nodes = interface.bbs_nodes
        add_channel(channel_name, channel_url, bbs_nodes, interface)
        send_message(f"Channel '{channel_name}' has been added to the directory.", sender_id, interface)

    except Exception as e:
        logging.error(f"Error processing post channel command: {e}")
        send_message("Error processing post channel command.", sender_id, interface)


def handle_check_channel_command(sender_id, interface):
    try:
        channels = get_channels()
        if not channels:
            send_message("No channels available in the directory.", sender_id, interface)
            return

        response = "Available Channels:\n"
        for i, channel in enumerate(channels):
            response += f"{i + 1:02d}. Name: {channel[1]}\n"
        response += "\nPlease reply with the number of the channel you want to view."
        send_message(response, sender_id, interface)

        update_user_state(sender_id, {'command': 'CHECK_CHANNEL', 'step': 1, 'channels': channels})

    except Exception as e:
        logging.error(f"Error processing check channel command: {e}")
        send_message("Error processing check channel command.", sender_id, interface)


def handle_read_channel_command(sender_id, message, state, interface):
    try:
        channels = state.get('channels', [])
        message_number = int(message) - 1

        if message_number < 0 or message_number >= len(channels):
            send_message("Invalid channel number. Please try again.", sender_id, interface)
            return

        channel_id, channel_name, channel_url = channels[message_number]
        response = f"Channel Name: {channel_name}\nChannel URL: {channel_url}"
        send_message(response, sender_id, interface)

        update_user_state(sender_id, None)

    except ValueError:
        send_message("Invalid input. Please enter a valid channel number.", sender_id, interface)
    except Exception as e:
        logging.error(f"Error processing read channel command: {e}")
        send_message("Error processing read channel command.", sender_id, interface)


def handle_list_channels_command(sender_id, interface):
    try:
        channels = get_channels()
        if not channels:
            send_message("No channels available in the directory.", sender_id, interface)
            return

        response = "Available Channels:\n"
        for i, channel in enumerate(channels):
            response += f"{i+1:02d}. Name: {channel[1]}\n"
        response += "\nPlease reply with the number of the channel you want to view."
        send_message(response, sender_id, interface)

        update_user_state(sender_id, {'command': 'LIST_CHANNELS', 'step': 1, 'channels': channels})

    except Exception as e:
        logging.error(f"Error processing list channels command: {e}")
        send_message("Error processing list channels command.", sender_id, interface)


def handle_quick_help_command(sender_id, interface):
    # Load quick commands config for dynamic help
    quick_config = configparser.ConfigParser()
    quick_config.read('config.ini')

    # Default command info
    cmd_info = {
        'send_mail': {'default': 'SM,,', 'desc': 'Send Mail'},
        'check_mail': {'default': 'CM', 'desc': 'Check Mail'},
        'post_bulletin': {'default': 'PB,,', 'desc': 'Post Bulletin'},
        'check_bulletin': {'default': 'CB,,', 'desc': 'Check Bulletins'},
        'post_channel': {'default': 'CHP,,', 'desc': 'Post Channel'},
        'list_channels': {'default': 'CHL', 'desc': 'List Channels'},
        'last_users': {'default': 'LU', 'desc': 'Last Users'},
    }

    response = "✈️QUICK COMMANDS✈️\n"

    for cmd_name, info in cmd_info.items():
        prefix = info['default']
        enabled = True

        if 'quick_commands' in quick_config and cmd_name in quick_config['quick_commands']:
            value = quick_config['quick_commands'][cmd_name]
            parts = value.rsplit(',', 1)
            if len(parts) == 2:
                prefix = parts[0].upper()
                enabled = parts[1].strip().lower() == 'true'

        if enabled:
            response += f"{prefix} - {info['desc']}\n"

    send_message(response, sender_id, interface)


def handle_last_users_command(sender_id, interface):
    """Display the last 3 users who interacted with the BBS."""
    users = get_last_users(3)
    if not users:
        response = "📋Last Users📋\nNo recent users."
    else:
        response = "📋Last Users📋\n"
        for short_name, long_name, last_activity in users:
            response += f"{short_name} [{long_name}]: {last_activity}\n"
    send_message(response, sender_id, interface)


# AI Chat handlers

# Global chat handler instance (initialized on first use)
_chat_handler = None

# An AI turn takes seconds (LLM rounds, a web search). Turns run here, one at
# a time in the order the messages came, so the meshtastic thread that
# delivers every packet stays free for the menus meanwhile.
_ai_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ai-chat')

AI_CHAT_MENU = "🤖AI Chat🤖\nSend message. I can do mail, bulletins, mesh status, web search.\ne[X]it, [R]eset, [H]elp"


def _get_chat_handler():
    """Get or initialize the chat handler."""
    global _chat_handler
    if _chat_handler is None:
        try:
            # Read LLM config
            llm_config = configparser.ConfigParser()
            llm_config.read('config.ini')

            if 'llm' not in llm_config:
                logging.warning("LLM configuration not found in config.ini")
                return None

            api_key = llm_config['llm'].get('api_key', '')
            model = llm_config['llm'].get('model', 'gpt-4o-mini')
            max_reply_messages = llm_config['llm'].getint('max_reply_messages', 3)
            reasoning_effort = llm_config['llm'].get('reasoning_effort', '').strip() or None

            if not api_key:
                logging.warning("LLM API key not configured")
                return None

            from llm_chat import ChatHandler
            _chat_handler = ChatHandler(api_key, model, max_reply_messages, reasoning_effort)
            logging.info(f"LLM Chat initialized with model: {model}")

        except Exception as e:
            logging.error(f"Failed to initialize LLM Chat: {e}")
            return None

    return _chat_handler


def handle_ai_command(sender_id, interface):
    """Handle the initial AI Chat menu selection."""
    chat_handler = _get_chat_handler()

    if chat_handler is None:
        send_message("AI Chat not available. Check config.", sender_id, interface)
        update_user_state(sender_id, None)
        return

    send_message(AI_CHAT_MENU, sender_id, interface)
    update_user_state(sender_id, {'command': 'AI_CHAT', 'step': 1})


def handle_ai_steps(sender_id, message, step, state, interface):
    """Handle AI chat conversation steps."""
    message = message.strip()
    message_lower = message.lower()

    # Handle single-letter commands
    if message_lower == 'x':
        send_message("Exiting AI Chat.", sender_id, interface)
        handle_help_command(sender_id, interface)
        return

    if message_lower == 'h':
        handle_ai_command(sender_id, interface)
        return

    # The first message after a confirmation answers it. The change goes
    # with that message now, so an X or H sent after it can't drop it.
    pending = _take_pending(sender_id)
    # Everything else, R and the answers to confirmations included, waits its turn
    _ai_worker.submit(_ai_turn, sender_id, message, interface, time.monotonic(), pending)


def stop_ai_chat():
    """Drop the AI turns still waiting, so the process can exit after the current one."""
    _ai_worker.shutdown(wait=False, cancel_futures=True)


def _in_ai_chat(state):
    return bool(state) and state.get('command') == 'AI_CHAT'


def _take_pending(sender_id, asked_only=True):
    """
    The change waiting for the user's Y, removed from their state.

    Returns None if there is none, or (asked_only) if its confirmation
    hasn't gone out yet: a message from before can't answer it.
    """
    taken = []

    def take(state):
        pending = state.get('pending') if _in_ai_chat(state) else None
        if pending is None or (asked_only and pending.asked is None):
            return state
        taken.append(pending)
        return {key: value for key, value in state.items() if key != 'pending'}

    change_user_state(sender_id, take)
    return taken[0] if taken else None


def _store_pending(sender_id, pending):
    """Keep a change for the user's Y, unless they have left AI chat meanwhile. Returns whether it was kept."""
    kept = []

    def store(state):
        if _in_ai_chat(state):
            kept.append(pending)
            return {**state, 'pending': pending}
        return state

    change_user_state(sender_id, store)
    return bool(kept)


def _answer_pending(sender_id, message, pending, confirmed, received, interface):
    """Run or cancel the change the user was asked about: code only, no LLM."""
    if not confirmed:
        result = text_for('cancelled', pending.language)
    elif pending.expired(received):
        result = text_for('expired', pending.language)
    else:
        logging.info(f"AI Chat: {sender_id} confirmed: {pending.confirmation}")
        result = pending.run()
    add_conversation_message(str(sender_id), "user", message, pending.language)
    add_conversation_message(str(sender_id), "assistant", result, pending.language)
    send_message(result, sender_id, interface, numbered=True)


def _ai_turn(sender_id, message, interface, received, pending):
    """
    One AI chat message, on the AI worker.

    received is when the message came, and pending the change it answers,
    if any. The user may have left AI chat since.
    """
    try:
        if message.lower() == 'r':
            _take_pending(sender_id, asked_only=False)
            clear_conversation_history(str(sender_id))
            send_message("Conversation cleared.", sender_id, interface)
            if _in_ai_chat(get_user_state(sender_id)):
                send_message(AI_CHAT_MENU, sender_id, interface)
            return

        is_answer = answer_of(message) is not None
        if pending and is_answer:
            _answer_pending(sender_id, message, pending, answer_of(message), received, interface)
            return
        if is_answer and (get_user_state(sender_id) or {}).get('pending'):
            # Sent before the confirmation it would answer went out: a resend, or too soon
            logging.info(f"AI Chat: {sender_id} answered '{message}' before being asked, ignored")
            return

        chat_handler = _get_chat_handler()

        if chat_handler is None:
            send_message("AI Chat error. Try later.", sender_id, interface)
            change_user_state(sender_id, lambda state: None if _in_ai_chat(state) else state)
            return

        if pending:
            note = "The user didn't answer Y to your last confirmation, so that change was not made."
        elif is_answer:
            note = "No change is waiting for the user's Y/N, so none was made."
        else:
            note = None

        turn = chat_handler.process_message(
            sender_num=sender_id,
            message=message,
            interface=interface,
            get_history_func=get_conversation_history,
            add_message_func=add_conversation_message,
            note=note
        )

        for text in turn.outbox:
            send_message(text, sender_id, interface, numbered=True)
        if turn.pending:
            if not _store_pending(sender_id, turn.pending):
                logging.info(f"AI Chat: {sender_id} left, not asking: {turn.pending.confirmation}")
                return
            send_message(turn.reply, sender_id, interface, numbered=True)
            turn.pending.asked = time.monotonic()
            return
        if turn.reply:
            send_message(turn.reply, sender_id, interface, numbered=True)
        for done in turn.after_reply:
            done()

    except Exception:
        # A Future would keep the exception to itself
        logging.exception(f"AI Chat error for {sender_id}")
        send_message("Error. Try again.", sender_id, interface)


# Telegram bridge: the mesh side. telegram_bridge.py runs the bot; server.py
# sets it here when TELEGRAM_BOT_TOKEN is set, else T says it is off.

_bridge = None

TELEGRAM_OFF = "Telegram bridge is not configured."
TELEGRAM_NO_CONTACTS = "📱Telegram📱\nNo contacts yet. [I]nvite someone, e[X]it"
TELEGRAM_CHAT_MENU = "📱Telegram: {title}\nWrite here. [L]ist [I]nvite [P]osition [H]elp e[X]it"
TELEGRAM_LIST_END = "Number to open, D+number to remove, e[X]it"


def set_telegram_bridge(bridge):
    global _bridge
    _bridge = bridge


def _node_id_of(sender_id):
    return f'!{sender_id:08x}'


def _contact_title(contact):
    return contact.name if contact.tg_user_id is not None else f'{contact.name} (waiting)'


def _own_contact(contact_id, node_id):
    """The node's contact with this id; None if it is gone or another node's."""
    contact = get_telegram_contact(contact_id) if contact_id is not None else None
    return contact if contact is not None and contact.node_id == node_id else None


def _telegram_state(sender_id, step, contact, **extra):
    update_user_state(sender_id, {'command': 'TELEGRAM_CHAT', 'step': step,
                                  'contact_id': contact.id if contact else None, **extra})


def _open_telegram_chat(sender_id, interface, contact):
    """Show a contact's chat and make it the one the user writes to; with no contact, the empty menu."""
    if contact is None:
        send_message(TELEGRAM_NO_CONTACTS, sender_id, interface)
    else:
        send_message(TELEGRAM_CHAT_MENU.format(title=_contact_title(contact)), sender_id, interface)
    _telegram_state(sender_id, 1, contact)


def handle_telegram_command(sender_id, interface):
    """T in the main menu: the chat with the contact used last."""
    if _bridge is None:
        send_message(TELEGRAM_OFF, sender_id, interface)
        update_user_state(sender_id, None)
        return
    purge_expired_telegram_invites()
    _open_telegram_chat(sender_id, interface, get_recent_telegram_contact(_node_id_of(sender_id)))


def handle_telegram_steps(sender_id, message, step, state, interface):
    """
    The Telegram menu. Step 1 is the chat: a message goes to the current
    contact, and single letters are commands. Step 2 is the list, step 3 asks
    the name for an invite, step 4 confirms a removal. process_message sends
    X, Q, B, U, A, T and the quick commands elsewhere before this runs.
    """
    if _bridge is None:
        handle_telegram_command(sender_id, interface)
        return
    message = message.strip()
    command = message.lower()
    node_id = _node_id_of(sender_id)
    contact = _own_contact(state.get('contact_id'), node_id)

    if command in ('h', ''):
        _open_telegram_chat(sender_id, interface, contact or get_recent_telegram_contact(node_id))
    elif step == 3:
        _telegram_invite(sender_id, message, interface, node_id, contact)
    elif step == 4:
        _telegram_remove(sender_id, message, state, interface, node_id, contact)
    elif command == 'l':
        _telegram_list(sender_id, interface, node_id, contact)
    elif command == 'i':
        send_message("Name for the contact? (e.g. wife)", sender_id, interface)
        _telegram_state(sender_id, 3, contact)
    elif command == 'p':
        share = not get_telegram_share_position(node_id)
        set_telegram_share_position(node_id, share)
        send_message(f"Position for Telegram contacts: {'on' if share else 'off'}", sender_id, interface)
    elif step == 2:
        _telegram_pick(sender_id, command, state, interface, node_id, contact)
    elif contact is None:
        if state.get('contact_id') is not None:
            send_message(RADIO_NOT_SENT.format(reason="that contact was removed"), sender_id, interface)
        _open_telegram_chat(sender_id, interface, get_recent_telegram_contact(node_id))
    elif contact.tg_user_id is None:
        send_message(RADIO_NOT_SENT.format(reason=f"{contact.name} hasn't joined yet"), sender_id, interface)
    else:
        # Nothing comes back on success: the radio already showed BBS0's ACK
        _bridge.send_to_telegram(node_id, contact.id, message)


def _telegram_list(sender_id, interface, node_id, current):
    """L: the contacts, numbered; * marks the one the user writes to."""
    purge_expired_telegram_invites()
    contacts = get_telegram_contacts(node_id)
    if not contacts:
        _open_telegram_chat(sender_id, interface, None)
        return
    lines = [f"{i} {c.name} ({c.tg_name if c.tg_user_id is not None else 'waiting'})"
             f"{'*' if current is not None and c.id == current.id else ''}"
             for i, c in enumerate(contacts, 1)]
    send_message('\n'.join(lines + [TELEGRAM_LIST_END]), sender_id, interface)
    _telegram_state(sender_id, 2, current, contacts=[c.id for c in contacts])


def _telegram_pick(sender_id, command, state, interface, node_id, current):
    """Step 2: a number opens that contact, D and a number asks to remove it."""
    remove = command.startswith('d')
    number = command[1:].strip() if remove else command
    listed = state.get('contacts') or []
    picked = None
    if number.isdecimal() and 1 <= int(number) <= len(listed):
        picked = _own_contact(listed[int(number) - 1], node_id)
    if picked is None:
        send_message(f"Not sent. {TELEGRAM_LIST_END}", sender_id, interface)
    elif remove:
        send_message(f"Remove {picked.name}? Y/N", sender_id, interface)
        _telegram_state(sender_id, 4, current, remove_id=picked.id)
    else:
        touch_telegram_contact(picked.id)   # T opens it next time
        send_message(f"Telegram: {_contact_title(picked)}", sender_id, interface)
        _telegram_state(sender_id, 1, picked)


def _telegram_remove(sender_id, message, state, interface, node_id, current):
    """Step 4: Y removes the contact, and a linked one is told on Telegram; anything else keeps it."""
    target = _own_contact(state.get('remove_id'), node_id)
    if target is None:
        send_message("Already removed.", sender_id, interface)
    elif answer_of(message) is True:
        _bridge.remove_contact(target)
        send_message(f"{target.name} removed.", sender_id, interface)
        if current is not None and current.id == target.id:
            current = None
    else:
        send_message("Not removed.", sender_id, interface)
    _telegram_state(sender_id, 1, current or get_recent_telegram_contact(node_id))


def _telegram_invite(sender_id, message, interface, node_id, current):
    """Step 3: the name for a new contact, then the invite to hand them."""
    purge_expired_telegram_invites()
    name = message
    problem = contact_name_problem(name)
    if problem is None and any(c.name.casefold() == name.casefold() for c in get_telegram_contacts(node_id)):
        problem = f"You already have {name}. Another name, or e[X]it"
    if problem is None and _bridge.username is None:
        send_message("Telegram is unreachable now. Try again later.", sender_id, interface)
        _telegram_state(sender_id, 1, current)
        return
    if problem is None:
        try:
            contact_id, code = _bridge.create_invite(node_id, name)
        except ValueError:
            problem = f"You already have {name}. Another name, or e[X]it"
    if problem is not None:
        send_message(problem, sender_id, interface)
        return
    # Written to the Telegram person, so the mesh user can forward it as it is
    days = _bridge.settings.invite_ttl_days
    send_message(f"Chat with me through this bot:\n{_bridge.invite_link(code)}\n"
                 f"Valid {days} day{'' if days == 1 else 's'}, one use.", sender_id, interface)
    _telegram_state(sender_id, 1, get_telegram_contact(contact_id))
