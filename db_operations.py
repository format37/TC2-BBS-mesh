import logging
import sqlite3
import threading
import uuid
from collections import namedtuple
from datetime import datetime, timedelta, timezone

from meshtastic import BROADCAST_NUM

from utils import (
    send_bulletin_to_bbs_nodes,
    send_delete_bulletin_to_bbs_nodes,
    send_delete_mail_to_bbs_nodes,
    send_mail_to_bbs_nodes, send_message, send_channel_to_bbs_nodes
)


thread_local = threading.local()

def get_db_connection():
    if not hasattr(thread_local, 'connection'):
        thread_local.connection = sqlite3.connect('bulletins.db')
    return thread_local.connection

def initialize_database():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS bulletins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    board TEXT NOT NULL,
                    sender_short_name TEXT NOT NULL,
                    sender_id TEXT,
                    date TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    unique_id TEXT NOT NULL
                )''')
    # Migration: add sender_id column if missing (for existing databases)
    c.execute("PRAGMA table_info(bulletins)")
    columns = [col[1] for col in c.fetchall()]
    if 'sender_id' not in columns:
        c.execute("ALTER TABLE bulletins ADD COLUMN sender_id TEXT")
    c.execute('''CREATE TABLE IF NOT EXISTS mail (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender TEXT NOT NULL,
                    sender_short_name TEXT NOT NULL,
                    recipient TEXT NOT NULL,
                    date TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    content TEXT NOT NULL,
                    unique_id TEXT NOT NULL
                );''')
    c.execute('''CREATE TABLE IF NOT EXISTS channels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    url TEXT NOT NULL
                );''')
    c.execute('''CREATE TABLE IF NOT EXISTS ai_conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    language TEXT,
                    timestamp TEXT NOT NULL
                );''')
    c.execute('''CREATE TABLE IF NOT EXISTS user_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL UNIQUE,
                    short_name TEXT,
                    long_name TEXT,
                    last_activity TEXT NOT NULL
                );''')
    # The last bulletin each node saw as new, for "new since my last check" in AI chat
    c.execute('''CREATE TABLE IF NOT EXISTS bulletin_seen (
                    node_id TEXT PRIMARY KEY,
                    last_bulletin_id INTEGER NOT NULL
                );''')
    # The Telegram bridge. A contact is a link from a node to a Telegram user,
    # or, while tg_user_id is NULL, the invite that will make one.
    c.execute('''CREATE TABLE IF NOT EXISTS telegram_contacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    tg_user_id INTEGER,
                    tg_chat_id INTEGER,
                    tg_name TEXT,
                    invite_code TEXT,
                    invite_expires TEXT,
                    created TEXT NOT NULL,
                    last_used TEXT,
                    UNIQUE(node_id, tg_user_id)
                );''')
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS telegram_contacts_name "
              "ON telegram_contacts (node_id, name COLLATE NOCASE)")
    c.execute("CREATE UNIQUE INDEX IF NOT EXISTS telegram_contacts_code ON telegram_contacts (invite_code)")
    # The Telegram side's state: the link a Telegram user writes to
    c.execute('''CREATE TABLE IF NOT EXISTS telegram_users (
                    tg_user_id INTEGER PRIMARY KEY,
                    current_contact_id INTEGER,
                    language TEXT
                );''')
    # The mesh user's bridge settings
    c.execute('''CREATE TABLE IF NOT EXISTS telegram_nodes (
                    node_id TEXT PRIMARY KEY,
                    share_position INTEGER NOT NULL DEFAULT 1
                );''')
    conn.commit()
    print("Database schema initialized.")

def add_channel(name, url, bbs_nodes=None, interface=None):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO channels (name, url) VALUES (?, ?)", (name, url))
    conn.commit()

    if bbs_nodes and interface:
        send_channel_to_bbs_nodes(name, url, bbs_nodes, interface)


def get_channels():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, name, url FROM channels")
    return c.fetchall()


def delete_channel(channel_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM channels WHERE id = ?", (channel_id,))
    conn.commit()



def add_bulletin(board, sender_short_name, subject, content, bbs_nodes, interface, unique_id=None, sender_id=None):
    conn = get_db_connection()
    c = conn.cursor()
    date = datetime.now().strftime('%Y-%m-%d %H:%M')
    if not unique_id:
        unique_id = str(uuid.uuid4())
    c.execute(
        "INSERT INTO bulletins (board, sender_short_name, sender_id, date, subject, content, unique_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (board, sender_short_name, sender_id, date, subject, content, unique_id))
    conn.commit()
    if bbs_nodes and interface:
        send_bulletin_to_bbs_nodes(board, sender_short_name, subject, content, unique_id, bbs_nodes, interface)

    # New logic to send group chat notification for urgent bulletins
    if board.lower() == "urgent":
        notification_message = f"💥NEW URGENT BULLETIN💥\nFrom: {sender_short_name}\nTitle: {subject}\nDM 'CB,,Urgent' to view"
        send_message(notification_message, BROADCAST_NUM, interface)

    return unique_id


def get_bulletins(board):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, subject, sender_short_name, date, unique_id FROM bulletins WHERE board = ? COLLATE NOCASE", (board,))
    return c.fetchall()

def get_bulletin_content(bulletin_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT sender_short_name, date, subject, content, unique_id, sender_id FROM bulletins WHERE id = ?", (bulletin_id,))
    return c.fetchone()


def get_bulletin(bulletin_id):
    """One bulletin as (board, sender_short_name, sender_id, date, subject, content), or None."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT board, sender_short_name, sender_id, date, subject, content FROM bulletins WHERE id = ?", (bulletin_id,))
    return c.fetchone()


def get_bulletin_id(unique_id):
    """The local id of the bulletin with this unique_id, or None."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id FROM bulletins WHERE unique_id = ?", (unique_id,))
    result = c.fetchone()
    return result[0] if result else None


def get_latest_bulletins(board=None, limit=10):
    """
    The newest bulletins, newest first.

    Returns list of tuples: [(id, board, sender_short_name, date, subject), ...];
    board None means every board
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, board, sender_short_name, date, subject FROM bulletins "
        "WHERE ? IS NULL OR board = ? COLLATE NOCASE ORDER BY id DESC LIMIT ?",
        (board, board, limit)
    )
    return c.fetchall()


def get_bulletins_after(bulletin_id, board=None, limit=10):
    """
    Bulletins with an id above bulletin_id, oldest first.

    Returns list of tuples: [(id, board, sender_short_name, date, subject), ...];
    board None means every board
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT id, board, sender_short_name, date, subject FROM bulletins "
        "WHERE id > ? AND (? IS NULL OR board = ? COLLATE NOCASE) ORDER BY id LIMIT ?",
        (bulletin_id, board, board, limit)
    )
    return c.fetchall()


def count_bulletins(board=None, after_id=0):
    """How many bulletins have an id above after_id; board None means every board."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT COUNT(*) FROM bulletins WHERE id > ? AND (? IS NULL OR board = ? COLLATE NOCASE)",
        (after_id, board, board)
    )
    return c.fetchone()[0]


def get_bulletin_marker(node_id):
    """The id of the last bulletin node_id saw as new, or None if it never checked."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT last_bulletin_id FROM bulletin_seen WHERE node_id = ?", (node_id,))
    result = c.fetchone()
    return result[0] if result else None


def set_bulletin_marker(node_id, bulletin_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        """INSERT INTO bulletin_seen (node_id, last_bulletin_id) VALUES (?, ?)
           ON CONFLICT(node_id) DO UPDATE SET last_bulletin_id = excluded.last_bulletin_id""",
        (node_id, bulletin_id)
    )
    conn.commit()


def delete_bulletin(bulletin_id, bbs_nodes, interface):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT unique_id FROM bulletins WHERE id = ?", (bulletin_id,))
    result = c.fetchone()
    c.execute("DELETE FROM bulletins WHERE id = ?", (bulletin_id,))
    conn.commit()
    # Each BBS numbers bulletins its own way: peers find this one by its unique_id
    if result:
        send_delete_bulletin_to_bbs_nodes(result[0], bbs_nodes, interface)


def delete_bulletin_by_unique_id(unique_id):
    """Delete the bulletin a peer BBS deleted."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM bulletins WHERE unique_id = ?", (unique_id,))
    conn.commit()

def add_mail(sender_id, sender_short_name, recipient_id, subject, content, bbs_nodes, interface, unique_id=None):
    conn = get_db_connection()
    c = conn.cursor()
    date = datetime.now().strftime('%Y-%m-%d %H:%M')
    if not unique_id:
        unique_id = str(uuid.uuid4())
    c.execute("INSERT INTO mail (sender, sender_short_name, recipient, date, subject, content, unique_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
              (sender_id, sender_short_name, recipient_id, date, subject, content, unique_id))
    conn.commit()
    if bbs_nodes and interface:
        send_mail_to_bbs_nodes(sender_id, sender_short_name, recipient_id, subject, content, unique_id, bbs_nodes, interface)
    return unique_id

def get_mail(recipient_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, sender_short_name, subject, date, unique_id FROM mail WHERE recipient = ?", (recipient_id,))
    return c.fetchall()

def get_mail_content(mail_id, recipient_id):
    # TODO: ensure only recipient can read mail
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT sender_short_name, date, subject, content, unique_id FROM mail WHERE id = ? and recipient = ?", (mail_id, recipient_id,))
    return c.fetchone()

def delete_mail(unique_id, recipient_id, bbs_nodes, interface):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT recipient FROM mail WHERE unique_id = ?", (unique_id,))
        result = c.fetchone()
        if result is None:
            logging.error(f"No mail found with unique_id: {unique_id}")
            return  # Early exit if no matching mail found
        recipient_id = result[0]
        logging.info(f"Attempting to delete mail with unique_id: {unique_id} by {recipient_id}")
        c.execute("DELETE FROM mail WHERE unique_id = ? and recipient = ?", (unique_id, recipient_id,))
        conn.commit()
        send_delete_mail_to_bbs_nodes(unique_id, bbs_nodes, interface)
        logging.info(f"Mail with unique_id: {unique_id} deleted and sync message sent.")
    except Exception as e:
        logging.error(f"Error deleting mail with unique_id {unique_id}: {e}")
        raise


def get_sender_id_by_mail_id(mail_id):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT sender FROM mail WHERE id = ?", (mail_id,))
    result = c.fetchone()
    if result:
        return result[0]
    return None


# AI Conversation functions

def add_conversation_message(sender_id, role, content, language=None):
    """Add a message to the AI conversation history."""
    conn = get_db_connection()
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    c.execute(
        "INSERT INTO ai_conversations (sender_id, role, content, language, timestamp) VALUES (?, ?, ?, ?, ?)",
        (str(sender_id), role, content, language, timestamp)
    )
    conn.commit()


def get_conversation_history(sender_id, limit=10):
    """
    Get conversation history for a sender.

    Returns list of dicts: [{"role": "user/assistant", "content": "..."}]
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT role, content FROM ai_conversations WHERE sender_id = ? ORDER BY id DESC LIMIT ?",
        (str(sender_id), limit)
    )
    rows = c.fetchall()
    # Reverse to get chronological order
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]


def clear_conversation_history(sender_id):
    """Clear all conversation history for a sender."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM ai_conversations WHERE sender_id = ?", (str(sender_id),))
    conn.commit()
    logging.info(f"Cleared AI conversation history for {sender_id}")


# User Activity functions

def record_user_activity(node_id, short_name, long_name):
    """Record or update user activity timestamp."""
    conn = get_db_connection()
    c = conn.cursor()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    c.execute(
        """INSERT INTO user_activity (node_id, short_name, long_name, last_activity)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(node_id) DO UPDATE SET
               short_name = excluded.short_name,
               long_name = excluded.long_name,
               last_activity = excluded.last_activity""",
        (str(node_id), short_name, long_name, timestamp)
    )
    conn.commit()


def get_last_users(limit=3):
    """
    Get the last N users who interacted with the BBS.

    Returns list of tuples: [(short_name, long_name, last_activity), ...]
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "SELECT short_name, long_name, last_activity FROM user_activity ORDER BY last_activity DESC LIMIT ?",
        (limit,)
    )
    return c.fetchall()


# Telegram bridge

TelegramContact = namedtuple('TelegramContact', [
    'id', 'node_id', 'name', 'tg_user_id', 'tg_chat_id', 'tg_name',
    'invite_code', 'invite_expires', 'created', 'last_used'])
_CONTACT_COLUMNS = ', '.join(TelegramContact._fields)


def utc_stamp(offset_seconds=0):
    """A UTC time as the Telegram tables keep it: text that sorts in time order."""
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).strftime('%Y-%m-%d %H:%M:%S.%f')


def _contacts(where, args):
    c = get_db_connection().cursor()
    c.execute(f"SELECT {_CONTACT_COLUMNS} FROM telegram_contacts WHERE {where}", args)
    return [TelegramContact(*row) for row in c.fetchall()]


def add_telegram_invite(node_id, name, code, expires):
    """
    A contact that waits for its invite code; returns its id.

    Raises sqlite3.IntegrityError when the node has a contact by that name or the code is taken.
    """
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO telegram_contacts (node_id, name, invite_code, invite_expires, created) VALUES (?, ?, ?, ?, ?)",
              (node_id, name, code, expires, utc_stamp()))
    conn.commit()
    return c.lastrowid


def get_telegram_contact(contact_id):
    found = _contacts("id = ?", (contact_id,))
    return found[0] if found else None


def get_telegram_contacts(node_id):
    """A node's contacts, linked and waiting, in the order they were made."""
    return _contacts("node_id = ? ORDER BY id", (node_id,))


def get_recent_telegram_contact(node_id):
    """The contact T opens: the linked one used last, else the newest invite; None if the node has none."""
    found = _contacts("node_id = ? ORDER BY tg_user_id IS NULL, COALESCE(last_used, created) DESC, id DESC LIMIT 1",
                      (node_id,))
    return found[0] if found else None


def get_telegram_links(tg_user_id):
    """The contacts a Telegram user is linked through, one per node, in the order they were made."""
    return _contacts("tg_user_id = ? ORDER BY id", (tg_user_id,))


def find_telegram_invite(code, now=None):
    """The contact waiting for this invite code, if the code is valid at now (UTC stamp)."""
    found = _contacts("tg_user_id IS NULL AND invite_code = ? AND invite_expires > ?", (code, now or utc_stamp()))
    return found[0] if found else None


def redeem_telegram_invite(contact_id, code, tg_user_id, tg_chat_id, tg_name):
    """Link the contact waiting for code to a Telegram user; False if the invite is gone meanwhile."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute(
        "UPDATE telegram_contacts SET tg_user_id = ?, tg_chat_id = ?, tg_name = ?, invite_code = NULL, "
        "invite_expires = NULL, last_used = ? WHERE id = ? AND invite_code = ? AND tg_user_id IS NULL",
        (tg_user_id, tg_chat_id, tg_name, utc_stamp(), contact_id, code))
    conn.commit()
    return c.rowcount == 1


def touch_telegram_contact(contact_id):
    """A message went between the node and this contact: T opens it next."""
    conn = get_db_connection()
    conn.execute("UPDATE telegram_contacts SET last_used = ? WHERE id = ?", (utc_stamp(), contact_id))
    conn.commit()


def delete_telegram_contact(contact_id):
    """Remove a contact or invite. A Telegram user left with no link is forgotten."""
    contact = get_telegram_contact(contact_id)
    if contact is None:
        return
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM telegram_contacts WHERE id = ?", (contact_id,))
    c.execute("UPDATE telegram_users SET current_contact_id = NULL WHERE current_contact_id = ?", (contact_id,))
    if contact.tg_user_id is not None:
        c.execute("DELETE FROM telegram_users WHERE tg_user_id = ? AND NOT EXISTS "
                  "(SELECT 1 FROM telegram_contacts WHERE tg_user_id = ?)", (contact.tg_user_id, contact.tg_user_id))
    conn.commit()


def purge_expired_telegram_invites(now=None):
    """Delete the invites nobody redeemed in time; returns how many."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM telegram_contacts WHERE tg_user_id IS NULL AND invite_expires <= ?", (now or utc_stamp(),))
    conn.commit()
    return c.rowcount


def get_telegram_user(tg_user_id):
    """(current_contact_id, language) of a Telegram user; (None, None) for one the bridge doesn't know."""
    c = get_db_connection().cursor()
    c.execute("SELECT current_contact_id, language FROM telegram_users WHERE tg_user_id = ?", (tg_user_id,))
    return c.fetchone() or (None, None)


def set_telegram_current(tg_user_id, contact_id):
    """The link a Telegram user's messages go to."""
    conn = get_db_connection()
    conn.execute("INSERT INTO telegram_users (tg_user_id, current_contact_id) VALUES (?, ?) "
                 "ON CONFLICT(tg_user_id) DO UPDATE SET current_contact_id = excluded.current_contact_id",
                 (tg_user_id, contact_id))
    conn.commit()


def set_telegram_language(tg_user_id, language):
    conn = get_db_connection()
    conn.execute("INSERT INTO telegram_users (tg_user_id, language) VALUES (?, ?) "
                 "ON CONFLICT(tg_user_id) DO UPDATE SET language = excluded.language",
                 (tg_user_id, language))
    conn.commit()


def get_telegram_share_position(node_id):
    """Whether the node's Telegram contacts may ask where it is (on unless the node switched it off)."""
    c = get_db_connection().cursor()
    c.execute("SELECT share_position FROM telegram_nodes WHERE node_id = ?", (node_id,))
    row = c.fetchone()
    return bool(row[0]) if row else True


def set_telegram_share_position(node_id, share):
    conn = get_db_connection()
    conn.execute("INSERT INTO telegram_nodes (node_id, share_position) VALUES (?, ?) "
                 "ON CONFLICT(node_id) DO UPDATE SET share_position = excluded.share_position",
                 (node_id, int(share)))
    conn.commit()
