import os
import secrets
import subprocess
import math
import numpy as np

MAX_TURNS = 100
FOLDER = 'training_data/temp'
N = 5

map_sizes = [32, 40, 48, 56, 64]
print("Generating data")
while True:
    map_size = secrets.choice(map_sizes)

    seed = np.random.random_integers(1, 100000)
    commands = [
        f'activate halite && halite.exe --seed {seed} --replay-directory replays/ --no-timeout --no-replay --no-logs --width {map_size} --height {map_size} --turn-limit {MAX_TURNS} "python MyBot.py" "python MyBot.py"',
        f'activate halite && halite.exe --seed {seed} --replay-directory replays/ --no-timeout --no-replay --no-logs --width {map_size} --height {map_size} --turn-limit {MAX_TURNS} "python MyBot.py" "python MyBot.py" "python MyBot.py" "python MyBot.py"']

    command = secrets.choice(commands)
    for i in range(N):
        subprocess.run(f'activate halite && {command} && deactivate', shell=True)

    games = os.listdir(FOLDER)
    sorted(games, key=lambda x: int(x.split("-")[0]))

    print(f"Picking top {math.floor(len(games)/3)} games")
    print(games[-math.floor(len(games)/3):])
    for w in games[-math.floor(len(games)/3):]:
        file_path = os.path.join(FOLDER, w)
        game = np.load(file_path)
        halite_amount = int(w.split("-")[0])
        np.save(f"training_data/{w}", game)

    for the_file in os.listdir(FOLDER):
        file_path = os.path.join(FOLDER, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)