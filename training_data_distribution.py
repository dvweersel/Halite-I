import os
import numpy as np
import matplotlib.pyplot as plt

all_files = os.listdir('training_data')

halite_amounts = []

for f in all_files:
    halite_amount = int(f.split("-")[0])

    halite_amounts.append(halite_amount)

print(len(halite_amounts))

plt.hist(halite_amounts, bins=10)
plt.show()

print(np.mean(halite_amounts))