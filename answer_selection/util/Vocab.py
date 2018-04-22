# NON-debugged code

import sys
import numpy as np

class Vocab(object):
  """docstring for Data"""

  def __init__(self, path):
    self.size = 0
    self._index = {}
    self._tokens = {}
    file = open(path, "r")
    while True:
      line = file.readline().rstrip('\n')
      if line == "":
        break
      self._tokens[self.size] = line
      self._index[line] = self.size
      self.size = self.size + 1
    file.close()

    unks = ['<unk>', '<UNK>', 'UUUNKKK']
    for tok in unks:
      self.unk_index = self._index[tok] if tok in self._index else None
      if self.unk_index:
        self.unk_token = tok
        break
  
    starts = ['<s>', '<S>']
    for tok in starts:
      self.start_index = self._index[tok] if tok in self._index else None
      if self.start_index:
        self.start_token = tok
        break
  
    ends = ['</s>', '</S>']
    for tok in ends:
      self.end_index = self._index[tok] if tok in self._index else None
      if self.end_index:
        self.end_token = tok
        break

  def contains(self, w):
    result = False
    if w in self._index:
     result = True
    return result

  def add(self, w):
    if w in self._index:
      return self._index[w]
    self._tokens[self.size] = w
    self._index[w] = self.size
    self.size = self.size + 1
    return self.size
  
  def index(self, w):
    index = self._index[w] if w in self._index else None
    if index is None:
      print(w)
      if self.unk_index is None:
         print('Token not in vocabulary and no UNK token defined: ' + w) # TODO0: change to error
         sys.exit()
      return self.unk_index
    return index

  def token(self, i):
    if i < 0 or i > self.size:
      print('Index ' + i + ' out of bounds') # TODO0: change to error
      sys.exit()
    return self._tokens[i]

  def map(self, tokens):
    length = len(tokens)
    output = np.zeros(length)   #torch.IntTensor(length)
    for i in range(0, length):
      output[i] = self.index(tokens[i])
    return output

  def add_unk_token(self):
    if self.unk_token:
      self.unk_index = self.add('<unk>')

  def add_start_token(self):
    if self.start_token:
      self.start_index = self.add('<s>')

  def add_end_token(self):
    if self.end_token:
      self.end_index = self.add('</s>')
