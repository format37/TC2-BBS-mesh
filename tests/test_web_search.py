"""Perplexity Pro Search, with the HTTP layer mocked."""
import pytest
import requests

import web_search
from conftest import write_config

RESPONSE = {
    'id': 'resp_1', 'status': 'completed',
    'usage': {'cost': {'total_cost': 0.00158}},
    'output': [
        {'type': 'search_results', 'results': [
            {'id': 1, 'title': 'A', 'url': 'https://www.a.example/1'},
            {'id': 2, 'title': '', 'url': 'https://b.example/2'},
            {'id': 3, 'title': 'C', 'url': 'https://c.example/3'},
            {'id': 4, 'title': 'D', 'url': 'https://d.example/4'},
        ]},
        {'type': 'message', 'role': 'assistant', 'content': [
            {'type': 'output_text', 'text': 'Showers **tomorrow**, high 16°C [web:3][web:1] and windy [web:3]. [web:9]'},
        ]},
    ],
}


class FakeResponse:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f'{self.status_code} Client Error: Unauthorized for url: {web_search.API_URL}')

    def json(self):
        return self.data


class Calls(list):
    """The POSTs made, and the response they get."""


@pytest.fixture
def posts(monkeypatch):
    calls = Calls()
    calls.response = FakeResponse(RESPONSE)

    def post(url, json=None, headers=None, timeout=None):
        calls.append({'url': url, 'json': json, 'headers': headers, 'timeout': timeout})
        return calls.response

    monkeypatch.setattr(web_search.requests, 'post', post)
    return calls


def test_parse_strips_citations_and_markdown_and_names_the_cited_domains():
    text, domains = web_search.parse(RESPONSE)
    assert text == 'Showers tomorrow, high 16°C and windy.'
    assert domains == ['c.example', 'a.example']


def test_parse_without_citations_names_the_first_results():
    response = {'output': [RESPONSE['output'][0], {'type': 'message', 'content': [
        {'type': 'output_text', 'text': '# Forecast\n- rain\n* wind\nSee [the map](https://x.example).'}]}]}
    text, domains = web_search.parse(response)
    assert text == 'Forecast\nrain\nwind\nSee the map.'
    assert domains == ['a.example', 'b.example', 'c.example']


def test_search_is_pro_search_with_short_plain_answers(posts):
    answer = web_search.search('weather in Tbilisi tomorrow', 'pplx-test')
    assert answer == 'Showers tomorrow, high 16°C and windy.\nSources: c.example, a.example'
    [call] = posts
    assert call['url'] == 'https://api.perplexity.ai/v1/agent'
    assert call['json'] == {'preset': 'low', 'input': 'weather in Tbilisi tomorrow',
                            'instructions': web_search.INSTRUCTIONS}
    assert call['headers']['Authorization'] == 'Bearer pplx-test'
    assert call['timeout'] == 60


def test_search_failures_raise(posts):
    posts.response = FakeResponse({'error': {'message': 'bad key'}}, status=401)
    with pytest.raises(requests.HTTPError, match='401'):
        web_search.search('q', 'pplx-test')
    posts.response = FakeResponse({'status': 'failed', 'output': [], 'error': 'overloaded'})
    with pytest.raises(RuntimeError, match='no answer .status failed. overloaded'):
        web_search.search('q', 'pplx-test')


def test_the_key_comes_from_config(bbs_dir):
    assert web_search.api_key() == 'test-perplexity-key'
    write_config(bbs_dir, '[web_search]\nprovider = other\napi_key = x\n')
    assert web_search.api_key() == ''
    write_config(bbs_dir, '[llm]\napi_key = x\n')
    assert web_search.api_key() == ''
