import os
import secrets
import subprocess
import math
import numpy as np
import time

N = 5
TURN_LIMIT = 100

map_sizes = [32, 40, 48, 56, 64]
print("Generating data")
while True:
    map_size = secrets.choice(map_sizes)

    seed = np.random.random_integers(10000, 100000000)
    commands = [
        f'activate halite && halite.exe --seed {seed} --replay-directory replays/ --no-timeout --no-replay --no-logs --width {map_size} --height {map_size} --turn-limit {TURN_LIMIT} "python MyBot.py" "python MyBot.py"',
        f'activate halite && halite.exe --seed {seed} --replay-directory replays/ --no-timeout --no-replay --no-logs --width {map_size} --height {map_size} --turn-limit {TURN_LIMIT} "python MyBot.py" "python MyBot.py" "python MyBot.py" "python MyBot.py"']

    for i in range(N):
        command = secrets.choice(commands)
        subprocess.run(f'activate halite && {command} && deactivate', shell=True)