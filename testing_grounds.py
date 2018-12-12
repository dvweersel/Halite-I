import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import datetime as dt


df = pd.DataFrame(columns=['A', 'B'])

df.loc[1] = [1, 1]
df.loc[2] = [2, 1]
df.duplicated(subset=['A'])

# # Load model
# start = dt.datetime.now()
# model = tf.keras.models.load_model("models/phase1-1544124313")
# end = dt.datetime.now()
#
# print(f"It took {end-start} time")
#
# X = np.load(f"train_files/X-2.npy")
# y = np.load(f"train_files/y-2.npy")
#
# input = X[0]
#
# print(input.shape)
# pred = model.predict(input)
#
# choices = []
# for p in pred:
#     choices.append(np.argmax(p))
#
# plt.hist(choices)
# plt.show()



