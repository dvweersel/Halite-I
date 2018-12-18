import os
import secrets
import subprocess
import time
import numpy as np

MAX_TURNS = 100

map_sizes = [32, 40, 48, 56, 64]
print("Generating data")
while True:
    map_size = secrets.choice(map_sizes)

    seed = np.random.random_integers(1, 100000)
    for i in range(5):
        commands = [f'activate halite && halite.exe --seed {seed} --replay-directory replays/ --no-timeout --no-replay --no-logs --width {map_size} --height {map_size} --turn-limit {MAX_TURNS} "python MyBot.py" "python MyBot.py"',
                    f'activate halite && halite.exe --seed {seed} --replay-directory replays/ --no-timeout --no-replay --no-logs --width {map_size} --height {map_size} --turn-limit {MAX_TURNS} "python MyBot.py" "python MyBot.py" "python MyBot.py" "python MyBot.py"']

        command = secrets.choice(commands)
        subprocess.run(f'activate halite && {command} && deactivate', shell=True)

    games = os.listdir('training_data/temp')
    sorted(games, key=lambda x: int(x.split("-")[0]))

    for w in games[5:]:
        game = np.load(w)
        halite_amount = int(w.split("-")[0])
        np.save(f"training_data/{halite_amount}-{int(time.time()*1000)}.npy", game)