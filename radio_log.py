"""The radio's own log, kept in a file.

The firmware writes its debug log into the same USB stream as the API frames
(2.7.x with security.debug_log_api_enabled off), and meshtastic publishes each
line as meshtastic.log.line. It shows what the radio did with every packet:
the next hop it chose, its retransmissions, who relayed what, and the signal
of every packet it heard. That is how to tell whether a reply left by a relay
or was never picked up.

The lines go to radio.log in the working directory (the config directory in
Docker, so data/radio.log on the host), never to the BBS log, rotated so they
take at most MAX_BYTES * (BACKUPS + 1).
"""
import logging
import re
from logging.handlers import RotatingFileHandler

from pubsub import pub

PATH = 'radio.log'
MAX_BYTES = 20_000_000
BACKUPS = 4
TOPIC = 'meshtastic.log.line'

_ANSI = re.compile(r'\x1b\[[0-9;]*m')
logger = logging.getLogger('radio')
logger.propagate = False


def _on_line(line, interface=None):
    logger.info(_ANSI.sub('', line).rstrip())


def start(path=PATH, max_bytes=MAX_BYTES, backups=BACKUPS):
    """Write the radio's log lines to path from now on; returns the handler."""
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s.%(msecs)03d %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    pub.subscribe(_on_line, TOPIC)
    return handler


def stop(handler):
    pub.unsubscribe(_on_line, TOPIC)
    logger.removeHandler(handler)
    handler.close()
