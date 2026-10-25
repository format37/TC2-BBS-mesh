# TC²-BBS Meshtastic Version

This is the TC²-BBS system integrated with Meshtastic devices. The system allows for message handling, bulletin boards, mail systems, and a channel directory.

### Docker

If you're a Docker user, TC²-BBS Meshtastic is available on Docker Hub!

[![Docker HUB](https://icon-icons.com/downloadimage.php?id=151885&root=2530/PNG/128/&file=docker_button_icon_151885.png)](https://hub.docker.com/r/thealhu/tc2-bbs-mesh)

## Setup

### Requirements

- Python 3.x
- Meshtastic
- pypubsub
- openai and requests (for the AI Chat)

### Update and Install Git
   
   ```sh
   sudo apt update
   sudo apt upgrade
   sudo apt install git
   ```

### Installation

1. Clone the repository:
   
   ```sh
   cd ~
   git clone https://github.com/TheCommsChannel/TC2-BBS-mesh.git
   cd TC2-BBS-mesh
   ```

2. Set up a Python virtual environment:  
   
   ```sh
   python -m venv venv
   ```

3. Activate the virtual environment:  
   
   - On Windows:  
   
   ```sh
   venv\Scripts\activate  
   ```
   
   - On macOS and Linux:
   
   ```sh
   source venv/bin/activate
   ```

4. Install the required packages:  
   
   ```sh
   pip install -r requirements.txt
   ```

5. Rename `example_config.ini`:

   ```sh
   mv example_config.ini config.ini
   ```

6. Set up the configuration in `config.ini`:  

   You'll need to open up the config.ini file in a text editor and make your changes following the instructions below
   
   **[interface]**  
   If using `type = serial` and you have multiple devices connected, you will need to uncomment the `port =` line and enter the port of your device.   
   
   Linux Example:  
   `port = /dev/ttyUSB0`   
   
   Windows Example:  
   `port = COM3`   
   
   If using type = tcp you will need to uncomment the hostname = 192.168.x.x line and put in the IP address of your Meshtastic device.  
   
   **[sync]**  
   Enter a list of other BBS nodes you would like to sync messages and bulletins with. Separate each by comma and no spaces as shown in the example below.   
   You can find the nodeID in the menu under `Radio Configuration > User` for each node, or use this script for getting nodedb data from a device:  
   
   [Meshtastic-Python-Examples/print-nodedb.py at main · pdxlocations/Meshtastic-Python-Examples (github.com)](https://github.com/pdxlocations/Meshtastic-Python-Examples/blob/main/print-nodedb.py)  
   
   Example Config:  
   
   ```ini
   [interface]  
   type = serial  
   # port = /dev/ttyUSB0  
   # hostname = 192.168.x.x  
   
   [sync]  
   bbs_nodes = !f53f4abc,!f3abc123  
   ```

   **[llm]**  
   Turns on the AI Chat. Enter your OpenAI API key in `api_key`. `max_reply_messages` sets how many radio messages one AI reply may take (default 3). Reasoning models such as gpt-6-sol need `reasoning_effort = none` to call tools; leave it out for models without reasoning, such as gpt-4o-mini.

   **[web_search]**  
   Lets the AI Chat search the web with Perplexity Pro Search (about $0.0016 a search). Enter your Perplexity API key in `api_key`, or leave it empty to turn web search off.

   ```ini
   [llm]
   provider = openai
   model = gpt-6-sol
   reasoning_effort = none
   api_key = sk-proj-your-api-key-here
   max_reply_messages = 3

   [web_search]
   provider = perplexity
   api_key = pplx-your-api-key-here
   ```

### Running the Server

Run the server with:

```sh
python server.py
```

Be sure you've followed the Python virtual environment steps above and activated it before running.

### Running the Tests

```sh
pip install pytest
pytest tests
```

The tests use a fake radio and a scripted LLM: they never call the real APIs.

## Command line arguments
```
$ python server.py --help

████████╗ ██████╗██████╗       ██████╗ ██████╗ ███████╗
╚══██╔══╝██╔════╝╚════██╗      ██╔══██╗██╔══██╗██╔════╝
   ██║   ██║      █████╔╝█████╗██████╔╝██████╔╝███████╗
   ██║   ██║     ██╔═══╝ ╚════╝██╔══██╗██╔══██╗╚════██║
   ██║   ╚██████╗███████╗      ██████╔╝██████╔╝███████║
   ╚═╝    ╚═════╝╚══════╝      ╚═════╝ ╚═════╝ ╚══════╝
Meshtastic Version

usage: server.py [-h] [--config CONFIG] [--interface-type {serial,tcp}] [--port PORT] [--host HOST] [--mqtt-topic MQTT_TOPIC]

Meshtastic BBS system

options:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        System configuration file
  --interface-type {serial,tcp}, -i {serial,tcp}
                        Node interface type
  --port PORT, -p PORT  Serial port
  --host HOST           TCP host address
  --mqtt-topic MQTT_TOPIC, -t MQTT_TOPIC
                        MQTT topic to subscribe
```



## Automatically run at boot

If you would like to have the script automatically run at boot, follow the steps below:

1. **Edit the service file**
   
   First, edit the mesh-bbs.service file using your preferred text editor. The 3 following lines in that file are what we need to edit:
   
   ```sh
   User=pi
   WorkingDirectory=/home/pi/TC2-BBS-mesh
   ExecStart=/home/pi/TC2-BBS-mesh/venv/bin/python3 /home/pi/TC2-BBS-mesh/server.py
   ```
   
   The file is currently setup for a user named 'pi' and assumes that the TC2-BBS-mesh directory is located in the home directory (which it should be if the earlier directions were followed)
   
   We just need to replace the 4 parts that have "pi" in those 3 lines with your username.

2. **Configuring systemd**
   
   From the TC2-BBS-mesh directory, run the following commands:
   
   ```sh
   sudo cp mesh-bbs.service /etc/systemd/system/
   ```
   
   ```sh
   sudo systemctl enable mesh-bbs.service
   ```
   
   ```sh
   sudo systemctl start mesh-bbs.service
   ```
   
   The service should be started now and should start anytime your device is powered on or rebooted. You can check the status of the service by running the following command:
   
   ```sh
   sudo systemctl status mesh-bbs.service
   ```
   
   If you need to stop the service, you can run the following:
   
   ```sh
   sudo systemctl stop mesh-bbs.service
   ```
   
   If you need to restart the service, you can do so with the following command:
   
   ```sh
   sudo systemctl restart mesh-bbs.service
   ```

2. **Viewing Logs**

   Viewing past logs:
   ```sh
   journalctl -u mesh-bbs.service
   ```

   Viewing live logs:
   ```sh
   journalctl -u mesh-bbs.service -f
   ```

## Radio Configuration

Note: There have been reports of issues with some device roles that may allow the BBS to communicate for a short time, but then the BBS will stop responding to requests. 

The following device roles have been working: 
- **Client**
- **Router_Client**

The BBS hands the radio one message at a time: each message waits until the radio has answered the previous one (its ACK, or a NAK), for at most 30 seconds. Parts sent sooner collide on air with the receiver's ACK and are lost.

Radios on the ESP32-S3's own USB port (Heltec V4, LILYGO T-Beam Supreme and others with USB ID `303a:1001`) read from USB slowly on firmware before 2.8.1: the serial console sleeps up to 20 seconds at a time ([meshtastic/firmware#11864](https://github.com/meshtastic/firmware/issues/11864)). Messages still arrive, but each one can take up to about 20 seconds to go out. The fix, [meshtastic/firmware#11901](https://github.com/meshtastic/firmware/pull/11901), is in 2.8.1, an alpha as of October 2026, and is small enough to apply to a 2.7 build. With it, each further part of a reply goes out about 4 seconds after the previous one.

Before moving a BBS node to 2.8, note that 2.8 derives node IDs from the public key: the node's ID changes, so its users have to find it again.

## Features

- **Mail System**: Send and receive mail messages.
- **Bulletin Boards**: Post and view bulletins on various boards.
- **Channel Directory**: Add and view channels in the directory.
- **Statistics**: View statistics about nodes, hardware, and roles.
- **Wall of Shame**: View devices with low battery levels.
- **Fortune Teller**: Get a random fortune. Pulls from the fortunes.txt file. Feel free to edit this file remove or add more if you like.
- **AI Chat**: Talk to an AI assistant in plain language, in English, Russian or Georgian. See [AI Chat](#ai-chat).

## Usage

You interact with the BBS by sending direct messages to the node that's connected to the system running the Python script. Sending any message to it will get a response with the main menu.  
Make selections by sending messages based on the letter or number in brackets - Send M for [M]ail Menu for example.

### AI Chat

Send `A` from the main menu to start the AI Chat. Then write what you need, for example:

- "any new bulletins?" (new since your last check, on all boards)
- "check my mail", "read #12"
- "send mail to BOB: meet at 5"
- "post to News: the repeater is back"
- "delete bulletin #12"
- "how is the mesh?", "what channels are there?"
- "weather in Tbilisi tomorrow" (needs `[web_search]`)

Original mail and bulletin texts are sent to you word for word. Replies take at most `max_reply_messages` messages; a reply split over several messages is numbered `1/3`, `2/3`, ...

Nothing changes until you confirm. Before it sends mail, posts or deletes anything, the BBS asks, for example `Delete bulletin #12 "Party" (General)? Y/N`. Answer `Y` (or `да`, `კი`) to go ahead, `N` to cancel; anything else cancels and is taken as a new message. A confirmation expires after 10 minutes. You can delete only your own mail and bulletins (the admin node can delete any bulletin), and posting to Urgent follows `[allow_list]`.

Send `R` to reset the conversation, `H` for help, `X` to exit.

A video of it in use is available on our YouTube channel:

[![TC²-BBS-Mesh](https://img.youtube.com/vi/d6LhY4HoimU/0.jpg)](https://www.youtube.com/watch?v=d6LhY4HoimU)

## Thanks

**Meshtastic:**

Big thanks to [Meshtastic](https://github.com/meshtastic) and [pdxlocations](https://github.com/pdxlocations) for the great Python examples:

[python/examples at master · meshtastic/python (github.com)](https://github.com/meshtastic/python/tree/master/examples)

[pdxlocations/Meshtastic-Python-Examples (github.com)](https://github.com/pdxlocations/Meshtastic-Python-Examples)

**JS8Call:**

For the JS8Call side of things, big thanks to Jordan Sherer for JS8Call and the [example API Python script](https://bitbucket.org/widefido/js8call/src/js8call/tcp.py)

## License

GNU General Public License v3.0
