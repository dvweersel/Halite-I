import os

files = []
for f in os.listdir('training_data'):
    files.append(os.path.join('training_data', f))

print(len(files))