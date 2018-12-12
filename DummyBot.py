#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction

# This library allows you to generate random numbers.
import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("MyBot-2")

""" <<<Game Loop>>> """

direction_order = Direction.get_all_cardinals()

while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map
    command_queue = []

    for ship in me.get_ships():

        size = 16
        surroundings = []
        for y in range(-1*size, size+1):
            row = []
            for x in range(-1*size, size+1):
                row.append([x, y])

        # For each of your ships, move randomly if the ship is on a low halite location or the ship is full.
        #   Else, collect halite.
        if game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full:
            command_queue.append(
                ship.move(
                    random.choice([Direction.North, Direction.South, Direction.East, Direction.West])))
        else:
            command_queue.append(ship.stay_still())

    if len(me.get_ships()) < 1:
        if me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
            command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

