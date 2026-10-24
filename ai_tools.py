"""
Tools for the AI chat agent: the BBS features and web search.

build_tools() is called once per turn, for the user who sent the message.
The tools act with that user's rights, so none of them takes a user
argument. They never send anything themselves: original texts go to an
outbox, sent before the agent's reply.

A change (send, post, delete) doesn't run at once. Its tool checks the
arguments and permissions, then stores one pending action with a
confirmation written by code, and the turn ends. The action runs only when
the user answers Y, so text in mail, bulletins or web results can't act
with the user's rights.
"""

import json
import logging
import re
import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional

import web_search
from db_operations import (
    add_bulletin, add_mail, count_bulletins, delete_bulletin, delete_mail,
    get_bulletin, get_bulletin_id, get_bulletin_marker, get_bulletins_after,
    get_channels, get_last_users, get_latest_bulletins, get_mail, get_mail_content,
    set_bulletin_marker
)
from utils import MAX_PAYLOAD_BYTES, get_node_id_from_num, send_message, truncate

BOARDS = ('General', 'Info', 'News', 'Urgent')
LIST_LIMIT = 10
FIRST_CHECK = 5          # bulletins a user who never checked gets as new
PENDING_SECONDS = 600    # how long a change waits for the user's Y
LOW_BATTERY = 20

YES = {'y', 'yes', 'да', 'д', 'კი', 'ki'}
NO = {'n', 'no', 'нет', 'н', 'არა', 'ara'}

# What the user reads around a change, by language; English is the fallback
TEXTS = {
    'send_mail': {
        'english': 'Send mail to {to} "{subject}": {body}? Y/N',
        'russian': 'Отправить {to} письмо «{subject}»: {body}? Да/Нет',
        'georgian': 'გავუგზავნო {to}-ს წერილი „{subject}“: {body}? კი/არა',
    },
    'delete_mail': {
        'english': 'Delete mail #{id} "{subject}" from {sender}? Y/N',
        'russian': 'Удалить письмо #{id} «{subject}» от {sender}? Да/Нет',
        'georgian': 'წავშალო {sender}-ის წერილი #{id} „{subject}“? კი/არა',
    },
    'post_bulletin': {
        'english': 'Post to {board} "{subject}": {body}? Y/N',
        'russian': 'Опубликовать в {board} «{subject}»: {body}? Да/Нет',
        'georgian': 'გამოვაქვეყნო {board}-ზე „{subject}“: {body}? კი/არა',
    },
    'post_urgent': {
        'english': 'Post to {board} "{subject}": {body}? It goes to the whole mesh. Y/N',
        'russian': 'Опубликовать в {board} «{subject}»: {body}? Получит вся сеть. Да/Нет',
        'georgian': 'გამოვაქვეყნო {board}-ზე „{subject}“: {body}? ამას მთელი ქსელი მიიღებს. კი/არა',
    },
    'delete_bulletin': {
        'english': 'Delete bulletin #{id} "{subject}" ({board})? Y/N',
        'russian': 'Удалить объявление #{id} «{subject}» ({board})? Да/Нет',
        'georgian': 'წავშალო განცხადება #{id} „{subject}“ ({board})? კი/არა',
    },
    'sent': {
        'english': 'Sent to {to}.',
        'russian': 'Отправлено {to}.',
        'georgian': 'წერილი გაეგზავნა {to}-ს.',
    },
    'posted': {
        'english': 'Posted #{id} to {board}.',
        'russian': 'Опубликовано #{id} в {board}.',
        'georgian': '{board}-ზე გამოქვეყნდა #{id}.',
    },
    'mail_deleted': {
        'english': 'Deleted mail #{id}.',
        'russian': 'Письмо #{id} удалено.',
        'georgian': 'წერილი #{id} წაიშალა.',
    },
    'bulletin_deleted': {
        'english': 'Deleted bulletin #{id}.',
        'russian': 'Объявление #{id} удалено.',
        'georgian': 'განცხადება #{id} წაიშალა.',
    },
    'gone': {
        'english': '#{id} is already gone.',
        'russian': '#{id} уже нет.',
        'georgian': '#{id} უკვე აღარ არსებობს.',
    },
    'cancelled': {
        'english': 'Cancelled.',
        'russian': 'Отменено.',
        'georgian': 'გაუქმდა.',
    },
    'expired': {
        'english': 'Too late: that request expired. Ask again.',
        'russian': 'Поздно: запрос устарел. Попросите снова.',
        'georgian': 'მოთხოვნას დადასტურების ვადა გაუვიდა. სცადე თავიდან.',
    },
}

logger = logging.getLogger("AI-Tools")


def text_for(key, language, **fields):
    templates = TEXTS[key]
    return templates.get(language, templates['english']).format(**fields)


def confirmation(key, language, **fields):
    """A confirmation on one line that fits one message: the body, then the subject, are cut to make room."""
    fields = {name: ' '.join(str(value).split()) for name, value in fields.items()}
    if 'body' in fields:
        # The template asks its own question: "at 5.? Y/N" would read badly
        fields['body'] = fields['body'].rstrip('.!?') or fields['body']
    text = text_for(key, language, **fields)
    for name in ('body', 'subject'):
        overflow = len(text.encode('utf-8')) - MAX_PAYLOAD_BYTES
        if overflow > 0 and name in fields:
            fields[name] = truncate(fields[name], max(len(fields[name].encode('utf-8')) - overflow, 12), '…')
            text = text_for(key, language, **fields)
    return text


def answer_of(message):
    """True when the message is a yes, False when it is a no, None for anything else."""
    word = message.strip().lower().rstrip('.!')
    if word in YES:
        return True
    if word in NO:
        return False
    return None


@dataclass
class PendingAction:
    """A change the agent asked for, waiting for the user's Y."""
    confirmation: str          # what the user is asked, written by code
    run: Callable[[], str]     # makes the change; returns what to tell the user
    language: str
    asked: Optional[float] = None   # time.monotonic() once the confirmation went out

    def expired(self, at):
        """Whether an answer that came at `at` is too late."""
        return at - self.asked > PENDING_SECONDS


class ToolError(Exception):
    """Why a tool can't do what the model asked; the model gets it as the result."""


def _quote(text):
    """Text from other people or the web, marked as data for the model."""
    text = text.replace('<<<', '< < <').replace('>>>', '> > >')
    return f'<<<\n{text}\n>>>'


def _id(args, name):
    try:
        return int(str(args.get(name)).strip().lstrip('#'))
    except ValueError:
        raise ToolError(f'{name} must be a number, like 12')


def _text(args, name):
    value = args.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f'{name} is missing')
    return value.strip()


def _flag(args, name, default):
    value = args.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in ('true', 'yes', '1')
    return bool(value)


def _board(value):
    for board in BOARDS:
        if str(value or '').strip().lower() == board.lower():
            return board
    raise ToolError(f'board must be one of {", ".join(BOARDS)}')


def _bulletin_lines(rows):
    return '\n'.join(f'#{id} {board} {date} {sender}: {subject}' for id, board, sender, date, subject in rows)


def _describe(node_id, user):
    return f"{user.get('shortName', '?')} ({user.get('longName', '')}, {node_id})"


def _schema(name, description, properties=None, required=()):
    return {'type': 'function', 'function': {
        'name': name,
        'description': description,
        'parameters': {'type': 'object', 'properties': properties or {}, 'required': list(required)},
    }}


_SHOW_ORIGINAL = {
    'type': 'boolean',
    'description': 'True (default): the original text goes to the user as is. '
                   'False: only you get it, to summarize, translate or answer about it.',
}
_CONFIRMED = 'The user confirms with Y/N; never ask yourself.'

SCHEMAS = {
    'list_mail': _schema(
        'list_mail', f"List the user's mail, newest first (at most {LIST_LIMIT}): #id, date, sender, subject."),
    'read_mail': _schema(
        'read_mail', "Read one of the user's mails.",
        {'mail_id': {'type': 'integer'}, 'show_original': _SHOW_ORIGINAL}, ['mail_id']),
    'send_mail': _schema(
        'send_mail', f'Send mail from the user to another node. {_CONFIRMED}',
        {'to': {'type': 'string', 'description': "The recipient's short name as is, like BOB, or a node id, like !1a2b3c4d"},
         'subject': {'type': 'string', 'description': 'A short subject'},
         'body': {'type': 'string', 'description': 'The message'}},
        ['to', 'subject', 'body']),
    'delete_mail': _schema(
        'delete_mail', f"Delete one of the user's mails. {_CONFIRMED}",
        {'mail_id': {'type': 'integer'}}, ['mail_id']),
    'list_bulletins': _schema(
        'list_bulletins', f'List bulletins, newest first (at most {LIST_LIMIT}): #id, board, date, sender, subject.',
        {'board': {'type': 'string', 'enum': list(BOARDS), 'description': 'One board; omit for all'},
         'new_only': {'type': 'boolean',
                      'description': "Only those new since the user's last check. On all boards, "
                                     "this also marks them seen."}}),
    'read_bulletin': _schema(
        'read_bulletin', 'Read one bulletin.',
        {'bulletin_id': {'type': 'integer'}, 'show_original': _SHOW_ORIGINAL}, ['bulletin_id']),
    'post_bulletin': _schema(
        'post_bulletin', f'Post a bulletin as the user. Urgent also alerts the whole mesh. {_CONFIRMED}',
        {'board': {'type': 'string', 'enum': list(BOARDS)},
         'subject': {'type': 'string', 'description': 'A short subject'},
         'body': {'type': 'string'}},
        ['board', 'subject', 'body']),
    'delete_bulletin': _schema(
        'delete_bulletin', f'Delete a bulletin: users may delete their own, the admin any. {_CONFIRMED}',
        {'bulletin_id': {'type': 'integer'}}, ['bulletin_id']),
    'list_channels': _schema(
        'list_channels', 'The channel directory: shared Meshtastic channels, with their names and URLs.'),
    'mesh_status': _schema(
        'mesh_status', 'Mesh status: nodes heard (all time, 24 h, 8 h, 1 h), the last 3 BBS users, '
                       f'nodes with battery under {LOW_BATTERY}%.'),
    'web_search': _schema(
        'web_search', 'Search the web, in a few seconds: news, weather, prices, anything current. '
                      'Ask one specific question, with the place and date when they matter.',
        {'query': {'type': 'string'}}, ['query']),
}
WRITES = {'send_mail', 'delete_mail', 'post_bulletin', 'delete_bulletin'}


class AgentTools:
    """The tools of one turn, acting as the user who sent the message."""

    def __init__(self, sender_num, interface, language='english', search_key=''):
        self.sender_num = sender_num
        self.interface = interface
        self.node_id = get_node_id_from_num(sender_num, interface) or f'!{sender_num:08x}'
        self.language = language
        self.search_key = search_key
        self.outbox = []        # texts for the user, sent before the reply
        self.pending = None     # the change waiting for the user's Y
        self.after_reply = []   # what to do once the user has the reply
        self._tools = {
            'list_mail': self.list_mail,
            'read_mail': self.read_mail,
            'send_mail': self.send_mail,
            'delete_mail': self.delete_mail,
            'list_bulletins': self.list_bulletins,
            'read_bulletin': self.read_bulletin,
            'post_bulletin': self.post_bulletin,
            'delete_bulletin': self.delete_bulletin,
            'list_channels': self.list_channels,
            'mesh_status': self.mesh_status,
        }
        if search_key:
            self._tools['web_search'] = self.search_web

    @property
    def names(self):
        return list(self._tools)

    @property
    def schemas(self):
        """The tools, in the OpenAI Chat Completions format."""
        return [SCHEMAS[name] for name in self._tools]

    @property
    def user(self):
        """The user's short and long name, as the mesh knows them."""
        user = (self.interface.nodes.get(self.node_id) or {}).get('user') or {}
        return _describe(self.node_id, user) if user else self.node_id

    def call(self, name, arguments):
        """Run the tool the model called. Returns its result as text, errors included."""
        tool = self._tools.get(name)
        if tool is None:
            return f'Error: there is no tool {name}.'
        try:
            args = json.loads(arguments or '{}')
        except ValueError:
            args = None
        if not isinstance(args, dict):
            return 'Error: the arguments must be a JSON object.'
        logger.info(f"{self.node_id} calls {name} {truncate(json.dumps(args, ensure_ascii=False), 200)}")
        if name in WRITES and self.pending is not None:
            return 'Error: one change at a time. The user is already asked to confirm another one.'
        try:
            return tool(args)
        except ToolError as e:
            return f'Error: {e}.'
        except Exception as e:
            logger.exception(f"Tool {name} failed")
            return f'Error: {name} failed ({type(e).__name__}).'

    # Reading

    def list_mail(self, args):
        mail = sorted(get_mail(self.node_id), reverse=True)
        if not mail:
            return 'The mailbox is empty.'
        heading = f'{len(mail)} in the mailbox' + (f', the newest {LIST_LIMIT}' if len(mail) > LIST_LIMIT else '')
        lines = '\n'.join(f'#{id} {date} from {sender}: {subject}'
                          for id, sender, subject, date, _ in mail[:LIST_LIMIT])
        return f'{heading}:\n{_quote(lines)}'

    def read_mail(self, args):
        mail_id = _id(args, 'mail_id')
        mail = get_mail_content(mail_id, self.node_id)
        if mail is None:
            raise ToolError(f"there is no mail #{mail_id} in the user's mailbox")
        sender, date, subject, content, _ = mail
        return self._original(f'Mail #{mail_id}\nFrom: {sender}\nDate: {date}\nSubject: {subject}\n\n{content}', args)

    def list_bulletins(self, args):
        board = _board(args['board']) if args.get('board') else None
        where = f' on {board}' if board else ''
        if not _flag(args, 'new_only', False):
            rows = get_latest_bulletins(board, LIST_LIMIT)
            if not rows:
                return f'No bulletins{where}.'
            total = count_bulletins(board)
            heading = f'{total} bulletins{where}' + (f', the newest {len(rows)}' if total > len(rows) else '')
            return f'{heading}:\n{_quote(_bulletin_lines(rows))}'

        # New since the last check. Only a check of every board moves the
        # marker, so bulletins on the other boards stay new.
        marker = get_bulletin_marker(self.node_id)
        if marker is None:
            rows = get_latest_bulletins(board, FIRST_CHECK)
            heading = f'First check, so the latest {len(rows)}{where}'
        elif board:
            rows = [row for row in get_latest_bulletins(board, LIST_LIMIT) if row[0] > marker]
            more = count_bulletins(board, after_id=marker) - len(rows)
            heading = f'{len(rows)} new{where} since the last check' + (
                f' ({more} older new ones not shown)' if more else '')
        else:
            # The oldest new ones first, so none is skipped when there are many
            rows = get_bulletins_after(marker, None, LIST_LIMIT)[::-1]
            more = count_bulletins(after_id=marker) - len(rows)
            heading = f'{len(rows)} new since the last check' + (
                f' ({more} more new: list them again for the rest)' if more else '')
        if board is None and (rows or marker is None):
            # Seen once the user has the reply, not when a turn that fails later read them
            self.after_reply.append(
                partial(set_bulletin_marker, self.node_id, max((row[0] for row in rows), default=0)))
        if not rows:
            return f'No new bulletins{where}.'
        return f'{heading}:\n{_quote(_bulletin_lines(rows))}'

    def read_bulletin(self, args):
        bulletin_id = _id(args, 'bulletin_id')
        bulletin = get_bulletin(bulletin_id)
        if bulletin is None:
            raise ToolError(f'there is no bulletin #{bulletin_id}')
        board, sender, _, date, subject, content = bulletin
        return self._original(
            f'Bulletin #{bulletin_id} ({board})\nFrom: {sender}\nDate: {date}\nSubject: {subject}\n\n{content}', args)

    def list_channels(self, args):
        channels = get_channels()
        if not channels:
            return 'The channel directory is empty.'
        lines = '\n'.join(f'{name}: {url}' for _, name, url in channels)
        return f'{len(channels)} channels:\n{_quote(lines)}'

    def mesh_status(self, args):
        nodes = list(self.interface.nodes.values())
        now = time.time()
        heard = [sum(1 for node in nodes if (node.get('lastHeard') or 0) >= now - seconds)
                 for seconds in (86400, 28800, 3600)]
        lines = [f'Nodes heard: {len(nodes)} in all, {heard[0]} in 24 h, {heard[1]} in 8 h, {heard[2]} in 1 h.']
        users = get_last_users(3)
        if users:
            lines.append('Last BBS users: ' + '; '.join(f'{short} ({long}) {when}' for short, long, when in users) + '.')
        low = []
        for node in nodes:
            level = (node.get('deviceMetrics') or {}).get('batteryLevel', 101)
            if level < LOW_BATTERY:
                user = node.get('user') or {}
                low.append(f"{user.get('shortName', '?')} ({user.get('longName', '')}) {level}%")
        lines.append(f'Battery under {LOW_BATTERY}%: ' + ', '.join(low) + '.' if low
                     else f'No node has battery under {LOW_BATTERY}%.')
        return _quote('\n'.join(lines))

    def search_web(self, args):
        query = _text(args, 'query')
        try:
            answer = web_search.search(query, self.search_key)
        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            raise ToolError(f'web search failed: {type(e).__name__}: {e}')
        return _quote(answer)

    def _original(self, text, args):
        """The text for the model, also sent to the user word for word when show_original is on."""
        if not _flag(args, 'show_original', True):
            return _quote(text)
        if text not in self.outbox:
            self.outbox.append(text)
        return ("Sent to the user word for word before your reply. Don't repeat, summarize or translate it "
                f"unless the user asked: a few words are enough, like \"Reply?\".\n{_quote(text)}")

    # Changing: each stores a pending action for the user's Y

    def send_mail(self, args):
        to, subject, body = _text(args, 'to'), _text(args, 'subject'), _text(args, 'body')
        recipient_id, recipient = self._find_node(to)
        recipient_name = recipient.get('shortName') or recipient_id
        # Anyone can take a short name: when it isn't unique, say which node
        namesakes = [node for node in list(self.interface.nodes.values())
                     if str((node.get('user') or {}).get('shortName', '')).lower() == recipient_name.lower()]
        if len(namesakes) > 1:
            recipient_name = _describe(recipient_id, recipient)

        def run():
            sender_short_name = self._short_name()
            add_mail(self.node_id, sender_short_name, recipient_id, subject, body,
                     self.interface.bbs_nodes, self.interface)
            notification_message = f"You have a new mail message from {sender_short_name}. Check your mailbox by responding to this message with CM."
            send_message(notification_message, recipient_id, self.interface)
            return text_for('sent', self.language, to=recipient_name)

        return self._propose(
            confirmation('send_mail', self.language, to=recipient_name, subject=subject, body=body), run)

    def delete_mail(self, args):
        mail_id = _id(args, 'mail_id')
        mail = get_mail_content(mail_id, self.node_id)
        if mail is None:
            raise ToolError(f"there is no mail #{mail_id} in the user's mailbox")
        sender, _, subject, _, unique_id = mail

        def run():
            if get_mail_content(mail_id, self.node_id) is None:
                return text_for('gone', self.language, id=mail_id)
            delete_mail(unique_id, self.node_id, self.interface.bbs_nodes, self.interface)
            return text_for('mail_deleted', self.language, id=mail_id)

        return self._propose(
            confirmation('delete_mail', self.language, id=mail_id, subject=subject, sender=sender), run)

    def post_bulletin(self, args):
        board, subject, body = _board(args.get('board')), _text(args, 'subject'), _text(args, 'body')
        allowed_nodes = self.interface.allowed_nodes
        if board == 'Urgent' and allowed_nodes and self.node_id not in allowed_nodes:
            raise ToolError('the user may not post to Urgent, only the nodes the BBS allows')

        def run():
            unique_id = add_bulletin(board, self._short_name(), subject, body, self.interface.bbs_nodes,
                                     self.interface, sender_id=self.node_id)
            return text_for('posted', self.language, id=get_bulletin_id(unique_id), board=board)

        key = 'post_urgent' if board == 'Urgent' else 'post_bulletin'
        return self._propose(confirmation(key, self.language, board=board, subject=subject, body=body), run)

    def delete_bulletin(self, args):
        bulletin_id = _id(args, 'bulletin_id')
        bulletin = get_bulletin(bulletin_id)
        if bulletin is None:
            raise ToolError(f'there is no bulletin #{bulletin_id}')
        board, _, owner_id, _, subject, _ = bulletin
        # Imported here: command_handlers imports this module
        from command_handlers import is_admin_or_owner
        if not is_admin_or_owner(self.sender_num, owner_id, self.interface):
            raise ToolError('the user can only delete their own bulletins')

        def run():
            if get_bulletin(bulletin_id) is None:
                return text_for('gone', self.language, id=bulletin_id)
            delete_bulletin(bulletin_id, self.interface.bbs_nodes or [], self.interface)
            return text_for('bulletin_deleted', self.language, id=bulletin_id)

        return self._propose(
            confirmation('delete_bulletin', self.language, id=bulletin_id, subject=subject, board=board), run)

    def _propose(self, confirmation_text, run):
        self.pending = PendingAction(confirmation_text, run, self.language)
        return 'The user is asked to confirm with Y/N.'

    def _short_name(self):
        user = (self.interface.nodes.get(self.node_id) or {}).get('user') or {}
        return user.get('shortName') or self.node_id

    def _find_node(self, to):
        """
        The (node id, user) the model means by to.

        An !id, else a short name, else a long name, else the one node whose
        names contain it. The user sees who it is in the confirmation.
        Raises a ToolError naming the candidates when it is unclear.
        """
        nodes = [(node_id, node['user']) for node_id, node in list(self.interface.nodes.items()) if node.get('user')]
        wanted = to.strip().strip('"\'«»„“').lstrip('@').lower()
        for node_id, user in nodes:
            if node_id.lower() == wanted:
                return node_id, user
        # Models write "!bob" too, and keep case endings: the Georgian "BOB-ს"
        name = re.sub(r"[-'’][^\x00-\x7f]+$", '', wanted.lstrip('!'))

        def names(user):
            return str(user.get('shortName', '')).lower(), str(user.get('longName', '')).lower()

        matches = [(node_id, user) for node_id, user in nodes if names(user)[0] == name]
        if len(matches) > 1:
            raise ToolError(f'{len(matches)} nodes have the short name {to}: '
                            + '; '.join(_describe(*match) for match in matches)
                            + '. Ask the user which one, then pass its !id')
        matches = matches or [(node_id, user) for node_id, user in nodes if names(user)[1] == name]
        matches = matches or [(node_id, user) for node_id, user in nodes if name and any(name in n for n in names(user))]
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ToolError(f'no node has the short name {to}. Similar: '
                            + '; '.join(_describe(*match) for match in matches[:5]))
        raise ToolError(f'no node has the short name {to}')


def build_tools(sender_num, interface, language='english'):
    """The tools for one turn of sender_num's AI chat; web_search only when it has a key."""
    return AgentTools(sender_num, interface, language, web_search.api_key())
