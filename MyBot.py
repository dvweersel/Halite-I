#!/usr/bin/env python3

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction, Position

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging
import numpy as np
import time
import secrets
import pandas as pd

import sys, os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.05)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
MODEL_NAME = 'gen1-0.0187-1546962535'
MODEL_EPOCH = 49

model = tf.keras.models.load_model(f"models/{MODEL_NAME}/{MODEL_NAME}-{MODEL_EPOCH}")
RANDOM_CHANCE = secrets.choice([0.15, 0.25, 0.35])

SIGHT_DISTANCE = 16
SAVE_THRESHOLD = 0

TOTAL_TURNS = 100
MAX_SHIPS = 99
RETURN_VALUE = 900

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
players = game.players

direction_order = Direction.get_all_cardinals() + [Direction.Still]
training_data = []

distance_map, initial_halite = game.game_map.return_map([game.me.shipyard.position])
logging.info(initial_halite)

logging.info("Loaded model")
# model.predict(np.ones((1,33,33,3)))

objectives = {}
game.ready("MyPythonBot")

""" <<<Game Loop>>> """
while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    # Data structures for making, storing and sending moves
    mission_control = pd.DataFrame(columns=['location', 'destination', 'move_id'])
    command_queue = []
    surroundings_dict = {}

    # Features
    dropoff_positions = [d.position for d in me.get_dropoffs() + [me.shipyard]]
    ship_positions = [s.position for s in me.get_ships()]
    total_ships = me.get_ship_amount()

    distance_map, avg_halite = game_map.return_map(dropoff_positions)

    for ship in me.get_ships():

        logging.info(ship)

        ship_id = ship.id
        # Determining objectives
        if ship.halite_amount > RETURN_VALUE:
            if ship.position == me.shipyard.position:
                objectives[ship_id] = 'm'
            else:
                objectives[ship_id] = 'r'
        else:
            objectives[ship_id] = 'm'

        size = SIGHT_DISTANCE
        surroundings = []
        for y in range(-1*size, size+1):
            row = []
            for x in range(-1*size, size+1):
                current_cell = game_map[ship.position + Position(x, y)]

                if current_cell.position in dropoff_positions:
                    drop_friend_foe = 1
                else:
                    drop_friend_foe = -1

                if current_cell.position in ship_positions:
                    ship_friend_foe = 1
                else:
                    ship_friend_foe = -1

                halite = round(current_cell.halite_amount/constants.MAX_HALITE, 2)
                a_ship = current_cell.ship
                structure = current_cell.structure

                if halite is None:
                    halite = 0
                if a_ship is None or a_ship.owner != me.id:
                    a_ship = 0
                else:
                    a_ship = round(a_ship.halite_amount/constants.MAX_HALITE, 2)

                if structure is None:
                    structure = 0
                else:
                    structure = drop_friend_foe

                amounts = (halite, a_ship, structure)
                row.append(amounts)

            surroundings.append(row)

        surroundings_dict[ship.id] = surroundings

        if objectives[ship_id] == 'r':
            logging.info("Returning")
            direction_choice = game_map.navigate_back(ship, distance_map)
        elif objectives[ship_id] == 'm':
            logging.info("Mining")
            if ship.halite_amount >= game_map[ship.position].halite_amount * constants.MOVE_COST_RATIO:
                if secrets.choice(range(int(1/RANDOM_CHANCE))) == 1:
                    direction_choice = secrets.choice(range(len(direction_order)))
                else:
                    prediction = model.predict([np.array(surroundings).reshape(-1, len(surroundings),
                                                                               len(surroundings), 3)])[0]
                    direction_choice = np.argmax(prediction)
                    logging.info(f"prediction: {direction_order[direction_choice]}")
            else:
                direction_choice = 4

        ship_destination = game_map.normalize(ship.position.directional_offset(direction_order[direction_choice]))

        # Save all relevant data
        mission_control.loc[ship.id] = [ship.position, ship_destination, direction_choice]

    if not mission_control.empty:
        while mission_control.destination.duplicated().any():
            collisions = mission_control[mission_control.duplicated(subset=['destination'], keep=False)]

            for c in collisions['destination'].unique():
                moves = collisions[collisions['destination'] == c]
                for idx, m in moves.iterrows():
                    if m['move_id'] == 4:
                        continue
                    else:
                        mission_control.loc[idx] = [m['location'], m['location'], 4]
                        break

        for ship_id in mission_control.index:
            ship = me.get_ship(ship_id)
            choice = mission_control.at[ship_id, 'move_id']

            # Only store relevant moves
            if objectives[ship_id] == 'm':
            	training_data.append([surroundings_dict[ship_id], choice])

            command_queue.append(ship.move(direction_order[choice]))

    if len(me.get_ships()) < MAX_SHIPS:
        if me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied \
                and not (mission_control['destination'] == me.shipyard.position).any():
            command_queue.append(me.shipyard.spawn())

    if game.turn_number == TOTAL_TURNS:
        avg_halite =  np.around(initial_halite/(game_map.width*game_map.height), 4)
        pct_gathered = np.around((me.halite_amount + (total_ships * constants.SHIP_COST))/initial_halite, 5)
        if pct_gathered >= SAVE_THRESHOLD:
            logging.info("Saving training data")
            np.save(f"training_data/{avg_halite}-{len(players)}-{pct_gathered}-{total_ships}-{int(time.time()*1000)}.npy", training_data)

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)