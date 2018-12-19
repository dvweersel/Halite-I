import os
import secrets
import subprocess
import time
import numpy as np
from multiprocessing import Pool

MAX_TURNS = 100
FOLDER = 'training_data/temp'
N = 2

def run_process(command):
    return os.system(f'activate halite && {command} && deactivate')

map_sizes = [32, 40, 48, 56, 64]
print("Generating data")

map_size = secrets.choice(map_sizes)

seed = np.random.random_integers(1, 100000)
commands = [
    f'activate halite && halite.exe --seed {seed} --replay-directory replays/ --no-timeout --no-replay --no-logs --width {map_size} --height {map_size} --turn-limit {MAX_TURNS} "python MyBot.py" "python MyBot.py"',
    f'activate halite && halite.exe --seed {seed} --replay-directory replays/ --no-timeout --no-replay --no-logs --width {map_size} --height {map_size} --turn-limit {MAX_TURNS} "python MyBot.py" "python MyBot.py" "python MyBot.py" "python MyBot.py"']

command_queu = []
for i in range(N):
    command = secrets.choice(commands)
    command_queu.append(f'activate halite && {command} && deactivate')
    # subprocess.run(f'activate halite && {command} && deactivate', shell=True)
print(command_queu)

# print(f'Running {N} parralel games')
pool = Pool(2)
pool.map(run_process, command_queu)