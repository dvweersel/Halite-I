import os
import secrets
import subprocess

MAX_TURNS = 100

map_sizes = [32, 40, 48, 56, 64]
print("Generating data")
while True:
    map_size = secrets.choice(map_sizes)

    commands = [f'activate halite && halite.exe --replay-directory replays/ --no-timeout --no-replay --no-logs --width {map_size} --height {map_size} --turn-limit {MAX_TURNS} "python MyBot.py" "python MyBot.py"',
                f'activate halite && halite.exe --replay-directory replays/ --no-timeout --no-replay --no-logs --width {map_size} --height {map_size} --turn-limit {MAX_TURNS} "python MyBot.py" "python MyBot.py" "python MyBot.py" "python MyBot.py"']

    command = secrets.choice(commands)
    subprocess.run(f'activate halite && {command} && deactivate', shell=True)
