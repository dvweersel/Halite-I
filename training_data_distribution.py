import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

all_files = os.listdir('training_data')

rows = []

for f in tqdm(all_files):
	avg_halite = float(f.split("-")[0])
	players = int(f.split("-")[1])
	gathered = float(f.split("-")[2])

	row = {'avg_halite' : avg_halite,
		'players' : players,
		'gathered': gathered}

	rows.append(row)

df = pd.DataFrame(rows, columns=['avg_halite', 'players', 'gathered'])

df.plot.scatter(x='avg_halite', y='gathered', c='players')

# # plt.hist(halite_amounts, bins=10)
# # plt.show()

# plt.scatter(halite_totals, halite_amounts)
# plt.show()
# print(np.mean(halite_amounts))