from flask import Flask, render_template, jsonify, request
import json
from zoombinis import *
app = Flask(__name__)

# config
app.config['DEBUG'] = True
app.config['TEMPLATES_AUTO_RELOAD'] = True

# maps ids to games
games = {}
id_count = 0


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/start')
def start():
    global games
    global id_count
    id_num = id_count
    id_count += 1
    games[id_num] = Game()
    zoombinis = games[id_num].zoombinis_json()
    data = {'id': id_num, 'zoombinis': zoombinis}
    return jsonify(data)

@app.route('/api/send', methods=['POST'])
def send():
    global games
    data = request.get_json()
    zoombini = data['zoombini']
    bridge = data['bridge']
    id_num = data['id']
    passed = games[id_num].send_zoombini(zoombini, bridge)
    game_over = not games[id_num].can_move()
    won = games[id_num].has_won()

    return jsonify({'passed': passed, 'game_over': game_over, 'won': won})



if __name__ == '__main__':
    app.run()