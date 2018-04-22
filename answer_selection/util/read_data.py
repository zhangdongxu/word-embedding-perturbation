# NON-debugged code

from Vocab import Vocab
import numpy as np

def read_embedding(vocab_path, emb_path):
  vocab = Vocab(vocab_path)
  embedding = np.load(emb_path)
  return vocab, embedding

def read_sentences(path, vocab, debug=False):
  sentences = []
  file = open(path, "r")
  if debug:
    count = 0
    while count<128:
    #while True:
      line = file.readline().rstrip('\n')
      if line == "":
        break
      tokens = line.split()  # TODO0: check if this splits str well?
      length = len(tokens)
      sent = np.zeros(max(length,3))
      counter = 0
      for i in range(0,length):
        token = tokens[i]
        sent[i] = int(vocab.index(token))
      if length < 3:
        for i in range(length, 3):
          sent[i] = int(vocab.index('unk')) # sent[len]
      if sent.sum() == 0:
        print('line: '+line)
      sentences.append(list(sent))
      count += 1
  else:
    count = 0
    while True:
    #while count<128:
      line = file.readline().rstrip('\n')
      if line == "":
        break
      tokens = line.split()  # TODO0: check if this splits str well?
      length = len(tokens)
      sent = np.zeros(max(length,3))
      counter = 0
      for i in range(0,length):
        token = tokens[i]
        sent[i] = int(vocab.index(token))
      if length < 3:
        for i in range(length, 3):
          sent[i] = int(vocab.index('unk')) # sent[len]
      if sent.sum() == 0:
        print('line: '+line)
      sentences.append(list(sent))
      count += 1
  file.close()
  return sentences

def read_relatedness_dataset(direc, vocab, debug=False):
  dataset = {}
  dataset['vocab'] = vocab
  file1 = 'a.toks'
  file2 = 'b.toks'
  dataset['lsents'] = read_sentences(direc + file1, vocab, debug)
  dataset['rsents'] = read_sentences(direc + file2, vocab, debug)
  dataset['size'] = len(dataset['lsents'])
  id_file = open(direc + 'id.txt', 'r')
  sim_file = open(direc + 'sim.txt')
  dataset['ids']= {}
  dataset['labels'] = []
  boundary_file = open(direc + 'boundary.txt')
  numrels_file = open(direc + 'numrels.txt')
  boundary = []
  counter = 0
  if debug:
    boundary = [0,34,48,64]
  else:   
    while True:
      line = boundary_file.readline().rstrip('\n')
      if line == "":
        break
      boundary.append(int(line))
  boundary_file.close()  
  dataset['boundary'] = boundary
  # read numrels data
  dataset['numrels'] = []
  for i in range(0, len(boundary)-1):
    tmp = numrels_file.readline().rstrip('\n')
    dataset['numrels'].append(int(tmp))
  numrels_file.close()

  for i in range(0, dataset['size']):
    dataset['ids'][i] = id_file.readline().rstrip('\n')
    tmp = int(sim_file.readline().rstrip('\n'))
    dataset['labels'].append(tmp) # twi and msp
  id_file.close()
  sim_file.close()
  
  return dataset
