
import pickle
import numpy as np


glove_dir="data/glove"
glove_pre="glove.840B"
glove_dim="300d"

path = glove_dir + '/' + glove_pre + '.' + glove_dim + '.txt'
vocabpath = glove_dir + '/' + glove_pre + '.vocab'
vecpath = glove_dir + '/' + glove_pre + '.' + glove_dim + '.npy'
print('Converting ' + path + ' to numpy pickled format')

# -- get dimension and number of lines
file = open(path, 'r')
count = 0
dim = 0
while True:
  line = file.readline()
  if line == '':
    break
  if count == 0:
    dim = len(line.split()) - 1
  count = count + 1
file.close()
print('count = %d' % count)
print('dim = %d' % dim)

# -- convert to numpy-friendly format
file = open(path, 'r')
vocab = open(vocabpath, 'w')
vecs = np.zeros([count, dim])
for i in range(0, count):
  tokens = file.readline().split()
  word = tokens[0]
  vocab.write(word + '\n')
  for j in range(0, dim):
    vecs[i, j] = float(tokens[j + 1])
file.close()
vocab.close()

np.save(vecpath, vecs)
