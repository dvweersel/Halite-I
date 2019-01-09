import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import seaborn as sns

all_files = os.listdir('training_data')

rows = []

for f in all_files:
	avg_halite = float(f.split("-")[0])
	players = int(f.split("-")[1])
	gathered = float(f.split("-")[2])

	row = {'avg_halite' : avg_halite,
		'players' : players,
		'gathered': gathered}

	rows.append(row)

df = pd.DataFrame(rows, columns=['avg_halite', 'players', 'gathered'])

training_file_names = []
for f in df.groupby('avg_halite').nlargest(2):
	training_file_names.append(os.path.join(TRAINING_DATA_DIR, f))

# g = sns.FacetGrid(df, col="players")
# g.map(plt.scatter, "avg_halite", "gathered", alpha=.7, edgecolor='black')
# g.add_legend()
# plt.show()

# df.to_pickle('report/gen1dataframe')
# # plt.hist(halite_amounts, bins=10)
# # plt.show()

# plt.scatter(halite_totals, halite_amounts)
# plt.show()
# print(np.mean(halite_amounts))