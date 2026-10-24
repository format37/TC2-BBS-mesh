"""Shared test setup.

command_handlers and message_processing read config.ini when they are
imported, and SQLite opens bulletins.db in the working directory. So the
suite starts in a scratch directory with a minimal config.ini, made here
before any test module imports them, and each test then gets a fresh
directory and database.
"""
import copy
import json
import os
import sys
import tempfile
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BBS_NUM = 0x0bb50001
ADMIN_ID = '!0a000001'

CONFIG = f"""\
[interface]
type = serial

[menu]
main_menu_items = Q, B, U, A, X
bbs_menu_items = M, B, C, X
utilities_menu_items = S, F, W, L, X

[admin]
admin_node = {ADMIN_ID}

[llm]
provider = openai
model = test-model
api_key = test-openai-key
max_reply_messages = 3

[web_search]
provider = perplexity
api_key = test-perplexity-key
"""


def write_config(directory, text=CONFIG):
    with open(os.path.join(directory, 'config.ini'), 'w', encoding='utf-8') as f:
        f.write(text)


_scratch = tempfile.mkdtemp(prefix='bbs-tests-')
write_config(_scratch)
os.chdir(_scratch)


class FakeInterface:
    """What the BBS uses of meshtastic's MeshInterface."""

    def __init__(self):
        self.nodes = {}
        self.myInfo = SimpleNamespace(my_node_num=BBS_NUM)
        self.bbs_nodes = []
        self.allowed_nodes = []
        self.sent = []          # (destination, text), in order
        self.oversize = []      # payloads the radio would reject

    def add_node(self, num, short_name, long_name=None, last_heard=None, battery=None):
        node_id = f'!{num:08x}'
        node = {'num': num, 'user': {'id': node_id, 'shortName': short_name,
                                     'longName': long_name or f'{short_name} node',
                                     'hwModel': 'HELTEC_V3', 'role': 'CLIENT'}}
        if last_heard is not None:
            node['lastHeard'] = last_heard
        if battery is not None:
            node['deviceMetrics'] = {'batteryLevel': battery}
        self.nodes[node_id] = node
        return num

    def sendText(self, text, destinationId, wantAck=False, wantResponse=False):
        if len(text.encode('utf-8')) > 200:
            self.oversize.append(text)
        self.sent.append((destinationId, text))
        return SimpleNamespace(id=len(self.sent))

    def texts(self, destination):
        return [text for dest, text in self.sent if dest == destination]


def _close_db():
    import db_operations
    if hasattr(db_operations.thread_local, 'connection'):
        db_operations.thread_local.connection.close()
        del db_operations.thread_local.connection


@pytest.fixture(autouse=True)
def bbs_dir(tmp_path, monkeypatch):
    """A fresh working directory, config.ini and database; sends don't sleep."""
    import db_operations
    import utils
    monkeypatch.chdir(tmp_path)
    write_config(tmp_path)
    _close_db()
    db_operations.initialize_database()
    monkeypatch.setattr(utils, 'time', SimpleNamespace(sleep=lambda seconds: None))
    utils.user_states.clear()
    yield tmp_path
    _close_db()


@pytest.fixture
def iface():
    interface = FakeInterface()
    yield interface
    assert not interface.oversize, f'payloads over 200 bytes: {interface.oversize}'


class FakeLLM:
    """
    A scripted stand-in for the OpenAI client: each completion takes the next step.

    A step is the reply text, a list of tool calls [(name, arguments), ...],
    or a function of the request that returns one of those.
    """

    def __init__(self, *steps):
        self.steps = list(steps)
        self.requests = []      # the arguments of each completion, as they were then
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(copy.deepcopy(request))
        assert self.steps, 'the LLM was called more times than scripted'
        step = self.steps.pop(0)
        if callable(step):
            step = step(request)
        if isinstance(step, str):
            message = {'role': 'assistant', 'content': step}
        else:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': f'call_{len(self.requests)}_{i}', 'type': 'function',
                 'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}}
                for i, (name, arguments) in enumerate(step)]}
        return ChatCompletion.model_validate({
            'id': 'chatcmpl-test', 'object': 'chat.completion', 'created': 0, 'model': request['model'],
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': message}],
        })

    def tool_results(self, request_index):
        """The tool results the model got in a request: [(tool_call_id, content), ...]"""
        return [(m['tool_call_id'], m['content']) for m in self.requests[request_index]['messages']
                if m['role'] == 'tool']


class InlineWorker:
    """Runs AI turns at once, on the calling thread."""

    def submit(self, fn, *args, **kwargs):
        fn(*args, **kwargs)


@pytest.fixture
def llm(monkeypatch):
    """The AI chat handler, on a scripted LLM (set llm.steps), with turns run inline."""
    import command_handlers
    from llm_chat import ChatHandler
    fake = FakeLLM()
    handler = ChatHandler('test-key', 'test-model', max_reply_messages=3)
    handler.client = fake
    monkeypatch.setattr(command_handlers, '_chat_handler', handler)
    monkeypatch.setattr(command_handlers, '_ai_worker', InlineWorker())
    return fake
