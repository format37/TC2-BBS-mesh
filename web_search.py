"""
Web search for the AI chat: Perplexity Pro Search, ported from pantherabot's research.py.

Sonar chat completions ended on 2026-09-27, so this calls the Agent API
(/v1/agent) with the `low` preset (Pro Search: a few search steps, seconds,
about $0.0016 a search). The answer comes back as plain text for the agent:
no citation markers, no markdown, and the domains of a few sources.
"""

import configparser
import logging
import re
import time
from urllib.parse import urlparse

import requests

API_URL = 'https://api.perplexity.ai/v1/agent'
PRESET = 'low'
TIMEOUT = 60
# The API takes instructions (checked 2026-10-24): answers come back short and plain
INSTRUCTIONS = 'Answer in at most 3 short plain-text sentences with the key facts. No markdown, no lists, no links.'
MAX_SOURCES = 3

# Presets cite as [web:3] (`fast` as [3]), the 3 being a search result's id
_CITE_RE = re.compile(r'\[(?:web:)?(\d+)\]')
_MARKDOWN = [
    (re.compile(r'\[([^\]]+)\]\([^)]*\)'), r'\1'),     # [text](url)
    (re.compile(r'\*\*|__|`'), ''),                     # bold, code
    (re.compile(r'^[ \t]*#{1,6}[ \t]*', re.M), ''),     # headings
    (re.compile(r'^[ \t]*[-*•][ \t]+', re.M), ''),      # list bullets
    (re.compile(r'[ \t]+'), ' '),
    (re.compile(r' (?=[.,;:!?])'), ''),                 # left before punctuation by a removed marker
    (re.compile(r'\n{3,}'), '\n\n'),
]

logger = logging.getLogger("Web-Search")


def api_key():
    """The Perplexity API key from config.ini, or '' when web search is off."""
    config = configparser.ConfigParser()
    config.read('config.ini')
    if config.get('web_search', 'provider', fallback='perplexity').strip().lower() != 'perplexity':
        return ''
    return config.get('web_search', 'api_key', fallback='').strip()


def parse(response):
    """
    The answer of an API response as plain text, and the domains of its sources.

    Returns:
        Tuple of (text, domains): up to MAX_SOURCES domains, the cited
        results first, else the first results
    """
    output = response.get('output') or []
    text = ''.join(
        part.get('text') or ''
        for item in output if item.get('type') == 'message'
        for part in item.get('content') or [] if part.get('type') == 'output_text'
    )

    urls = {}
    for item in output:
        if item.get('type') != 'search_results':
            continue
        for result in item.get('results') or []:
            if isinstance(result, dict) and result.get('url') and result.get('id') is not None:
                urls.setdefault(int(result['id']), result['url'])

    cited = [int(number) for number in _CITE_RE.findall(text) if int(number) in urls]
    domains = []
    for number in cited or sorted(urls):
        domain = urlparse(urls[number]).netloc.removeprefix('www.')
        if domain and domain not in domains:
            domains.append(domain)

    text = _CITE_RE.sub('', text)
    for pattern, replacement in _MARKDOWN:
        text = pattern.sub(replacement, text)
    return text.strip(), domains[:MAX_SOURCES]


def search(query, key):
    """
    Search the web and answer query.

    Returns:
        The answer as plain text, then a line with its source domains

    Raises:
        requests.RequestException, RuntimeError: when there is no answer
    """
    started = time.monotonic()
    response = requests.post(
        API_URL,
        json={'preset': PRESET, 'input': query, 'instructions': INSTRUCTIONS},
        headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    text, domains = parse(data)
    if not text:
        detail = data.get('error') or data.get('incomplete_details') or ''
        raise RuntimeError(f"no answer (status {data.get('status')}) {detail}".strip())

    cost = ((data.get('usage') or {}).get('cost') or {}).get('total_cost')
    logger.info(f"Search answered in {time.monotonic() - started:.1f}s: {len(text)} chars, cost {cost}")
    if domains:
        text += f"\nSources: {', '.join(domains)}"
    return text
