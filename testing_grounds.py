import os

games = os.listdir('training_data/temp')
games = sorted(games, key=lambda x: int(x.split("-")[0]))
print(games[4:])