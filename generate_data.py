import os
import secrets

MAX_TURNS = 100

map_sizes = [32, 40, 48, 56, 64]

os.system('call activate halite')
print("Generating data")
while True:
    map_size = secrets.choice(map_sizes)

    commands = [f'halite.exe --replay-directory replays/ --no-timeout --no-replay --no-logs --width {map_size} --height {map_size} --turn-limit {MAX_TURNS} "python MyBot.py" "python MyBot.py"',
                f'halite.exe --replay-directory replays/ --no-timeout --no-replay --no-logs --width {map_size} --height {map_size} --turn-limit {MAX_TURNS} "python MyBot.py" "python MyBot.py" "python MyBot.py" "python MyBot.py"']

    command = secrets.choice(commands)

    # os.system('halite.exe --replay-directory replays/ -vvv --width 32 --height 32 --turn-limit 50 "python MyBot.py" "python MyBot.py"')
    os.system(command)
