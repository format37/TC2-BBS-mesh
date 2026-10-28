#!/usr/bin/env python3

"""
TC²-BBS Server for Meshtastic by TheCommsChannel (TC²)
Date: 07/14/2024
Version: 0.1.6

Description:
The system allows for mail message handling, bulletin boards, and a channel
directory. It uses a configuration file for setup details and an SQLite3
database for data storage. Mail messages and bulletins are synced with
other BBS servers listed in the config.ini file.
"""

import logging
import os
import sys
import time

from command_handlers import set_telegram_bridge, stop_ai_chat
from config_init import initialize_config, get_interface, init_cli_parser, merge_config
from db_operations import initialize_database
from js8call_integration import JS8CallClient
from message_processing import on_receive
from pubsub import pub
from telegram_bridge import Bridge, Settings, TelegramApi
from utils import load_env_file

# General logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# JS8Call logging
js8call_logger = logging.getLogger('js8call')
js8call_logger.setLevel(logging.DEBUG)
js8call_handler = logging.StreamHandler()
js8call_handler.setLevel(logging.DEBUG)
js8call_formatter = logging.Formatter('%(asctime)s - JS8Call - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
js8call_handler.setFormatter(js8call_formatter)
js8call_logger.addHandler(js8call_handler)

def check_interface_health(interface):
    """Check if the meshtastic interface is still connected."""
    # Check if stream is still available
    if getattr(interface, 'stream', None) is None:
        return False

    # Check if stream is open (for serial)
    if hasattr(interface.stream, 'is_open') and not interface.stream.is_open:
        return False

    # Check reader thread
    rx_thread = getattr(interface, '_rxThread', None)
    if rx_thread and not rx_thread.is_alive():
        return False

    return True


def start_telegram_bridge(system_config, interface):
    """The Telegram bridge, running, when TELEGRAM_BOT_TOKEN is set; else None."""
    if not os.environ.get('TELEGRAM_BOT_TOKEN'):
        # Direct runs: a .env in the working directory or next to this file. Docker passes the variable itself.
        load_env_file('.env', os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
    token = os.environ.get('TELEGRAM_BOT_TOKEN', '').strip()
    if not token:
        logging.info("Telegram bridge is off: TELEGRAM_BOT_TOKEN is not set")
        return None
    bridge = Bridge(TelegramApi(token), interface, Settings.from_config(system_config['config']))
    set_telegram_bridge(bridge)
    bridge.start()
    return bridge


def stop_workers(bridge):
    stop_ai_chat()
    if bridge is not None:
        bridge.stop()


def display_banner():
    banner = """
████████╗ ██████╗██████╗       ██████╗ ██████╗ ███████╗
╚══██╔══╝██╔════╝╚════██╗      ██╔══██╗██╔══██╗██╔════╝
   ██║   ██║      █████╔╝█████╗██████╔╝██████╔╝███████╗
   ██║   ██║     ██╔═══╝ ╚════╝██╔══██╗██╔══██╗╚════██║
   ██║   ╚██████╗███████╗      ██████╔╝██████╔╝███████║
   ╚═╝    ╚═════╝╚══════╝      ╚═════╝ ╚═════╝ ╚══════╝
Meshtastic Version
"""
    print(banner)

def main():
    display_banner()
    args = init_cli_parser()
    config_file = None
    if args.config is not None:
        config_file = args.config
    system_config = initialize_config(config_file)

    merge_config(system_config, args)

    interface = get_interface(system_config)
    interface.bbs_nodes = system_config['bbs_nodes']
    interface.allowed_nodes = system_config['allowed_nodes']

    logging.info(f"TC²-BBS is running on {system_config['interface_type']} interface...")

    initialize_database()

    def receive_packet(packet, interface):
        on_receive(packet, interface)

    pub.subscribe(receive_packet, system_config['mqtt_topic'])

    bridge = start_telegram_bridge(system_config, interface)

    # Initialize and start JS8Call Client if configured
    js8call_client = JS8CallClient(interface)
    js8call_client.logger = js8call_logger

    if js8call_client.db_conn:
        js8call_client.connect()

    restart_delay = system_config.get('restart_delay', 30)

    try:
        while True:
            time.sleep(1)

            if not check_interface_health(interface):
                logging.error("Serial connection lost. Exiting for container restart...")
                logging.info(f"Waiting {restart_delay} seconds before exit...")
                stop_workers(bridge)
                time.sleep(restart_delay)
                interface.close()
                if js8call_client.connected:
                    js8call_client.close()
                sys.exit(1)

    except KeyboardInterrupt:
        logging.info("Shutting down the server...")
        stop_workers(bridge)
        interface.close()
        if js8call_client.connected:
            js8call_client.close()

if __name__ == "__main__":
    main()
