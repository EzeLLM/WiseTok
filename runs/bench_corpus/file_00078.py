import pickle
import socket
import sys
import time

from _thread import start_new_thread
from component import Card, Deck, CardPlaced, CARD_MAPPING, VAL_TO_CARD
GAME_DATA = {}
MESSAGE = ""
BUFFER_SIZE = 4096

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ip_address = '127.0.0.1'
port = 8081
server.connect((ip_address, port))


def debug_data(data: dict):
    print('='*20)
    print('======[DEBUG]======')

    def recur(data: dict):
        for key, val in data.items():
            if isinstance(val, dict):
                print(key)
                recur(val)
            else:
                print(key, val)
    recur(data)
    print('='*20)


def make_message(message, **kwargs):
    data = {}
    data['ID'] = CLIENT_ID
    data['NAME'] = name
    data['MSG'] = message
    data.update(kwargs)
    print(data)
    data = pickle.dumps(data)
    server.send(data)


def receive():
    global GAME_DATA, MESSAGE
    while True:
        data = server.recv(BUFFER_SIZE)
        data = pickle.loads(data)
        debug_data(data)
        MESSAGE = data['MSG']
        if(MESSAGE == "DONE"):
            print("Ranking:")
            print(data['WINNER'])
            print("Scoreboard:")
            print(data['SCOREBOARD'])
        GAME_DATA = data['GAME']


data = server.recv(BUFFER_SIZE)
CLIENT_ID = int(data.decode('utf-8'))

name = input("Name>>")
print(name, CLIENT_ID)
server.send(name.encode('utf-8'))

ready = input("Ready?>>")
make_message("READY")

data = server.recv(BUFFER_SIZE)
data = pickle.loads(data)
debug_data(data)
GAME_DATA = data['GAME']
MESSAGE = data['MSG']
start_new_thread(receive, ())
num_winner = len(GAME_DATA['winner'])
while True:
    if MESSAGE == "DONE":
        debug_data(data)
        sys.exit(0)
    if name not in GAME_DATA['winner']:
        if GAME_DATA['state'] == 'pick':
            print(MESSAGE)
            if name == GAME_DATA['turn']:
                if GAME_DATA['card_placed'].is_empty():
                    print('[PICK FREELY]')
                else:
                    num_cards = len(GAME_DATA['card_placed'])
                    last_val = VAL_TO_CARD[GAME_DATA['card_placed'].values[-1]]
                    print('[Curently there are {} cards of {}]'.format(num_cards, last_val))
                print()
                for j, card in enumerate(GAME_DATA['player_decks'][name].cards):
                    print('[{} {}]'.format(j, card), end=" ")
                print()
                selected = input("Select cards >>")
                selected = list(map(int, selected.split()))
                if(GAME_DATA['previous_card'] == None):
                    statement = input("Statement>>")
                else:
                    statement = GAME_DATA['previous_card']
                    print("Statement>>"+statement)
                make_message('PICK', SELECTED=selected, STATEMENT=statement)
            else:
                print("Now {} is picking cards".format(GAME_DATA['turn']))
            while GAME_DATA['state'] == 'pick':
                time.sleep(0.5)
        if GAME_DATA['state'] == 'lie_or_not':
            print(MESSAGE)
            if name != GAME_DATA['turn']:
                num_card = len(GAME_DATA['card_placed'].cards[-1])
                card_name = VAL_TO_CARD[GAME_DATA['card_placed'].values[-1]]
                msg = "player {} put {} {} cards".format(
                    GAME_DATA['turn'],
                    num_card,
                    card_name
                )
                print(msg)
                action = input("lie? [y/n]>>")
                if action == 'y':
                    action = True
                else:
                    action = False
                make_message('LIE', LIE=action)
            else:
                print("Wait for other players")
            while GAME_DATA['state'] == 'lie_or_not' and GAME_DATA['finish'] == False:
                time.sleep(0.5)
    else:
        print('[YOU WIN]')
        print(GAME_DATA['winner'])
        while len(GAME_DATA['winner']) == num_winner:
            time.sleep(0.5)
        num_winner = len(GAME_DATA['winner'])
