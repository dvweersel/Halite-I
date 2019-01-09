import tensorflow as tf
import os
import numpy as np
import time
import random
from tqdm import tqdm

import pandas as pd

import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from tensorflow.keras.callbacks import TensorBoard

import pickle

os.system('call activate halite')

LOAD_TRAIN_FILES = False # True if we have already batch train files
LOAD_PREV_MODEL = False
HALITE_THRESHOLD = 0.0187

TRAINING_CHUNK_SIZE = 200
PREV_MODEL_NAME = 'gen1-0.035-1545678705'
PREV_MODEL_EPOCH = 19
VALIDATION_GAME_COUNT = 0.1

PREV_MODEL_NAME_PATH = f"models/{PREV_MODEL_NAME}/{PREV_MODEL_NAME}-{PREV_MODEL_EPOCH}"
NAME = f"gen1-{HALITE_THRESHOLD}-{int(time.time())}"
EPOCHS = 50

TRAINING_DATA_DIR = 'training_data'

def createFolder(directory):
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
	except OSError:
		print ('Error: Creating directory. ' +  directory)

def cleanupFolder(directory):
	for the_file in os.listdir(directory):
		file_path = os.path.join(folder, the_file)
		try:
			if os.path.isfile(file_path):
				os.unlink(file_path)
		except Exception as e:
			print(e)        

def chunks(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in range(0, len(l), n):
		yield l[i:i + n]

def balance(x, y):
	_0 = []
	_1 = []
	_2 = []
	_3 = []
	_4 = []

	for x, y in zip(x, y):
		if y == 0:
			_0.append([x, y])
		elif y == 1:
			_1.append([x, y])
		elif y == 2:
			_2.append([x, y])
		elif y == 3:
			_3.append([x, y])
		elif y == 4:
			_4.append([x, y])

	shortest = min([len(_0),
					len(_1),
					len(_2),
					len(_3),
					len(_4)])

	_0 = _0[:shortest]
	_1 = _1[:shortest]
	_2 = _2[:shortest]
	_3 = _3[:shortest]
	_4 = _4[:shortest]

	balanced = _0 + _1 + _2 + _3 + _4

	random.shuffle(balanced)

	print(f"The shortest file was {shortest}, total balanced length is {len(balanced)}")

	xs = []
	ys = []

	for x, y in balanced:
		xs.append(x)
		ys.append(y)

	return xs, ys

training_file_names = []

rows = []
for f in os.listdir(TRAINING_DATA_DIR):
	avg_halite = float(f.split("-")[0])
	players = int(f.split("-")[1])
	gathered = float(f.split("-")[2])
	file = f

	row = {'avg_halite' : avg_halite,
		'players' : players,
		'gathered': gathered,
		'file' : f}

	rows.append(row)

df = pd.DataFrame(rows, columns=['avg_halite', 'players', 'gathered', 'file'])

games = df.sort_values('gathered', ascending=False).groupby('avg_halite', sort=False).head(5)

for f in games.values:
	training_file_names.append(os.path.join(TRAINING_DATA_DIR, f[3]))

print(f"After the threshold we have {len(training_file_names)}/{len(os.listdir(TRAINING_DATA_DIR))} games.")

random.shuffle(training_file_names)

tensorboard = TensorBoard(log_dir=f"logs/{NAME}")

# os.system(f'tensorboard --logdir="logs" --host localhost --port 8088')

TEST_LENGTH = math.floor(VALIDATION_GAME_COUNT*len(training_file_names))
if LOAD_TRAIN_FILES:
	X_test = np.load("train_files/X_test.npy")
	y_test = np.load("train_files/y_test.npy")
else:
	X_test = []
	y_test = []

	for f in tqdm(training_file_names[:TEST_LENGTH]):
		data = np.load(f)

		for d in data:
			X_test.append(np.array(d[0]))
			y_test.append(d[1])

	np.save("train_files/X_test.npy", X_test)
	np.save("train_files/y_test.npy", y_test)

# Convert for TF
X_test = np.array(X_test)

print('Created model and folders')

if LOAD_PREV_MODEL:
	print(f'Loading model {PREV_MODEL_NAME_PATH}')
	model = tf.keras.models.load_model(PREV_MODEL_NAME_PATH)
else:
	print(f'Creating model')

	model = Sequential()

	model.add(Conv2D(64, (3, 3), padding="same", input_shape=X_test.shape[1:]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

	model.add(Conv2D(64, (3, 3), padding="same"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

	model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

	model.add(Dense(64))

	model.add(Dense(5))
	model.add(Activation('softmax'))

opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-3)

model.compile(loss="sparse_categorical_crossentropy",
			  optimizer=opt,
			  metrics=['accuracy'])

X_test, y_test = balance(X_test, y_test)
createFolder(f"./models/{NAME}/")

for e in range(EPOCHS):
	training_file_chunks = chunks(training_file_names[TEST_LENGTH:], TRAINING_CHUNK_SIZE)

	print(f"Currently working on epoch {e+1}")

	for idx, training_files in enumerate(training_file_chunks):
		print(f"working on data chunk {idx+1}/{math.ceil(len(training_file_names)/TRAINING_CHUNK_SIZE)}")

		if LOAD_TRAIN_FILES or e > 0:
			X = np.load(f"train_files/X-{idx}.npy")
			y = np.load(f"train_files/y-{idx}.npy")
		else:
			X = []
			y = []

			for f in tqdm(training_files):
				data = np.load(f)

				for d in data:
					X.append(np.array(d[0]))
					y.append(d[1])

			X, y = balance(X, y)

			X = np.array(X)
			Y = np.array(y)

			X_test = np.array(X_test)
			y_test = np.array(y_test)

			np.save(f"train_files/X-{idx}.npy", X)
			np.save(f"train_files/y-{idx}.npy", y)

		model.fit(X, y, batch_size=32, epochs=1, validation_data=(X_test, y_test), callbacks=[tensorboard])

	model.save(f"models/{NAME}/{NAME}-{e}")

# os.system(f'tensorboard --logdir="./logs/{NAME}" --host localhost --port 8088')
print('Cleanup files')
cleanupFolder('train_files')
cleanupFolder('replays')