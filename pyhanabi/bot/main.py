import os
import argparse
import requests
from hanabi_client import HanabiClient
from bot_factory import BotFactory


# Authenticate, login to the Hanabi Live WebSocket server, and run forever
def main(username, password, agent):
    use_localhost = False

    # Get an authenticated cookie by POSTing to the login handler
    protocol = 'https'
    ws_protocol = 'wss'
    host = 'hanabi.marl-human-ai.com'

    path = '/login'
    ws_path = '/ws'
    url = protocol + '://' + host + path
    ws_url = ws_protocol + '://' + host + ws_path
    print(f'Authenticating to "{url}" with a username of "{username}".')
    resp = requests.post(
        url,
        {
            'username': username,
            'password': password,
            # This is normally the version of the JavaScript client,
            # but it will also accept "bot" as a valid version
            'version': 'bot',
        })

    # Handle failed authentication and other errors
    if resp.status_code != 200:
        print('Authentication failed:')
        print(resp.text)
        sys.exit(1)

    # Scrape the cookie from the response
    cookie = ''
    for header in resp.headers.items():
        if header[0] == 'Set-Cookie':
            cookie = header[1]
            break
    if cookie == '':
        print('Failed to parse the cookie from the authentication response '
              'headers:')
        print(resp.headers)
        sys.exit(1)

    HanabiClient(ws_url, cookie, agent)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--name", type=str, default="Bot-BR")
    parser.add_argument("--login_name", type=str, default=None)
    parser.add_argument("--password", type=str, default=None)
    args = parser.parse_args()
    if args.login_name is None:
        args.login_name = args.name

    agent = BotFactory[args.name]()
    main(args.login_name, args.password, agent)
