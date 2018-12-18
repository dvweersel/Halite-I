#!/usr/bin/env python3

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction, Position

# This library allows you to generate random numbers.
import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging

import os
import numpy as np
import time
import secrets

import pandas as pd

import sys, os
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
sys.stderr = stderr

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
SAVE_THRESHOLD = 5500
TOTAL_TURNS = 100
MAX_SHIPS = 10
SIGHT_DISTANCE = 16
RETURN_VALUE = 500

direction_order = Direction.get_all_cardinals() + [Direction.Still]

training_data = []

# Load model
model = tf.keras.models.load_model("models/phase2-6000-1544650113-15")
logging.info("Loaded model")
objectives = {}

distance_map, initial_halite = game.game_map.return_map([game.me.shipyard.position])
logging.info("Inital halite is: {}".format(initial_halite))

game.ready("MyPythonBot")

""" <<<Game Loop>>> """
while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    mission_control = pd.DataFrame(columns=['location', 'destination', 'move_id'])

    surroundings_dict = {}

    # Command queu to send moves
    command_queue = []

    dropoff_positions = [d.position for d in me.get_dropoffs() + [me.shipyard]]
    logging.info(dropoff_positions)
    ship_positions = [s.position for s in me.get_ships()]

    total_ships = me.get_ship_amount()

    distance_map, avg_halite = game_map.return_map(dropoff_positions)
    halite_collected = 1 - avg_halite / initial_halite
    logging.info("Halite distribution: avg: {}, collected {}".format(avg_halite, halite_collected))

    for ship in me.get_ships():

        logging.info(ship)

        if ship.halite_amount > RETURN_VALUE:
            if ship.position == me.shipyard.position:
                objectives[ship.id] = 'm'
            else:
                objectives[ship.id] = 'r'
        else:
            objectives[ship.id] = 'm'


        if objectives[ship.id] == 'r':
            direction_choice = game_map.navigate_back(ship, distance_map)
            logging.info(f'Navigating back in direction {direction_choice}')
        elif objectives[ship.id] == 'm':
            size = SIGHT_DISTANCE
            surroundings = []
            for y in range(-1*size, size+1):
                row = []
                for x in range(-1*size, size+1):
                    current_cell = game_map[ship.position + Position(x,y)]

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

            if ship.halite_amount >= game_map[ship.position].halite_amount * constants.MOVE_COST_RATIO:
                prediction = model.predict(np.expand_dims(surroundings, axis=0))
                direction_choice = np.argmax(prediction)

                # direction_choice = secrets.choice(range(len(direction_order)))
            else:
                direction_choice = 4
        else:
            logging.error('Error')

        ship_destination = game_map.normalize(ship.position.directional_offset(direction_order[direction_choice]))
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

            if objectives[ship.id] == 'm':
                training_data.append([surroundings_dict[ship_id], choice])

            command_queue.append(ship.move(direction_order[choice]))
            
            logging.info(ship)
            logging.info(f"Moving ship into {mission_control.loc[ship_id]['destination']}")

    if len(me.get_ships()) < MAX_SHIPS:
        if me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied \
                and not (mission_control['destination'] == me.shipyard.position).any():
            command_queue.append(me.shipyard.spawn())

    if game.turn_number == TOTAL_TURNS:
        halite_amount = total_ships*1000 + me.halite_amount
        if halite_amount >= SAVE_THRESHOLD:
            logging.info("Saving training data")
            np.save(f"training_data/{halite_amount}-{total_ships}-{int(time.time()*1000)}.npy", training_data)

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

