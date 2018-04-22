import numpy as np
import re
import itertools
from collections import Counter
import re

def process_sent(sent):
    sent = sent.strip()[1:-1]
    sent = re.sub(r'([\.,\?\(\)!":;])', r' \1 ', sent)
    sent = re.sub(r"('s)", r" \1 ", sent)
    sent = sent.strip().split()
    for i, w in enumerate(sent):
        if '<e1>' in w:
            e1_position = i
            sent[i].replace('<e1>', '')
        if '</e1>' in w:
            sent[i].replace('</e1>', '')
        if '<e2>' in w:
            e2_position = i
            sent[i].replace('<e2>', '')
        if '</e2>' in w:
            sent[i].replace('</e2>', '')
    e1_dist = [i - e1_position for i in range(len(sent))]
    e2_dist = [i - e2_position for i in range(len(sent))]
    sent = ' '.join(sent)
    return sent, e1_dist, e2_dist 

def count_label_num(data_file):
    rel2id = {}
    id2rel = {}
    count = 0
    for line in open(data_file):
        count += 1
        if count % 4 == 2:
            if line.strip() not in rel2id:
                rel2id[line.strip()] = len(rel2id)
                id2rel[len(id2rel)] = line.strip()
    return rel2id, id2rel

def load_data_and_labels(data_file, rel2id):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    if data_file == '':
        return None, None, None, None
    # Load data from files
    x_texts = []
    e1_dists = []
    e2_dists = []
    labels = []
    class_num = len(rel2id)
    count = 0
    for line in open(data_file):
        count += 1
        if count % 4 == 1:
            sent = line.split('\t')[1].strip()
            sent, e1_dist, e2_dist = process_sent(sent)
            x_texts.append(sent)
            e1_dists.append(e1_dist)
            e2_dists.append(e2_dist)
        if count % 4 == 2:
            rel = line.strip()
            rid = rel2id[rel]
            labels.append([1 if i==rid else 0 for i in range(class_num)])
    y = np.array(labels)
    e1_dists = np.array(e1_dists)
    e2_dists = np.array(e2_dists)
    return [x_texts, e1_dists, e2_dists, y]

def padding(e_dist, max_length):
    if e_dist == None:
        return None
    new_dist = []
    for batch in e_dist:
        new_dist.append(batch)
        if len(batch) < max_length:
            new_dist[-1].extend([0 for _ in range(max_length-len(batch))])
        elif len(batch) > max_length:
            new_dist[-1] = new_dist[-1][:max_length]
    return new_dist

def add_pad(x, e1_dist, e2_dist, filter_size):
    if x == None:
        return None, None, None
    pad = filter_size - 1
    new_x = np.zeros((x.shape[0], x.shape[1] + 2 * pad))
    new_x[:, pad:-pad] = x
    new_e1_dist = np.zeros((x.shape[0], x.shape[1] + 2 * pad))
    new_e1_dist[:, pad:-pad] = e1_dist
    new_e2_dist = np.zeros((x.shape[0], x.shape[1] + 2 * pad))
    new_e2_dist[:, pad:-pad] = e2_dist
    return new_x, new_e1_dist, new_e2_dist
    
def batch_iter(data, batch_size, embedding_dim, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def add_noise(sess, model, grad_noise, x, e1_dist, e2_dist, y, embedding_dim, random_type=None, word_keep=1.0, weight=0.0, replace_map = None):
    seq_length = len(x[0])
    batch_size = len(x)
    noise = np.ones([batch_size, seq_length, embedding_dim])
    
    #turn out to be bad.
    #force left and right paddings' word embedding to be zero.
    #for i in range(batch_size):
    #    noise[i,:4,:] = 0.0
    #    for j in range(seq_length)[::-1]:
    #        if x[i][j] != 0:
    #            noise[i,j+1:,:] = 0.0
        
    if random_type in ['Bernoulli', 'Bernoulli-semantic', 'Bernoulli-adversarial']:
        noise = noise / word_keep
    elif random_type in ['Gaussian', 'Gaussian-adversarial', 'Bernoulli-word', 'Bernoulli-idf', 'Bernoulli-polary', 'Replace']:
        pass
    
    if random_type == 'Adversarial':
        feed_dict = {
            model.input_x: x,
            model.e1_dist: e1_dist,
            model.e2_dist: e2_dist,
            model.input_y: y,
            model.noise: noise,
            model.dropout_keep_prob: 1.0
        }
        grad_noise_ = sess.run(grad_noise, feed_dict=feed_dict)
    elif random_type == 'Gaussian-adversarial':
        noise = np.random.normal(1, weight, [batch_size, seq_length, embedding_dim])
        feed_dict = {
            model.input_x: x,
            model.e1_dist: e1_dist,
            model.e2_dist: e2_dist,
            model.input_y: y,
            model.noise: noise,
            model.dropout_keep_prob: 1.0
        }
        grad_noise_ = sess.run(grad_noise, feed_dict=feed_dict)
    elif random_type == 'Bernoulli-adversarial':
        noise = np.random.choice(2,size=(batch_size, seq_length, embedding_dim), p=[1-word_keep, word_keep])
        feed_dict = {
            model.input_x: x,
            model.e1_dist: e1_dist,
            model.e2_dist: e2_dist,
            model.input_y: y,
            model.noise: noise,
            model.dropout_keep_prob: 1.0
        }
        grad_noise_ = sess.run(grad_noise, feed_dict=feed_dict)
        number_change = (1-word_keep) * seq_length * embedding_dim
    
    for bi in range(batch_size):
        if random_type == 'Bernoulli':
            noise[bi,:,:] = np.random.choice(2,size=(seq_length, embedding_dim), p=[1-word_keep, word_keep])
        if random_type == 'Gaussian':
            noise[bi,:,:] = np.random.normal(1, weight, [seq_length, embedding_dim])
        if random_type == 'Adversarial':
            grad_noise_[bi] /= (np.linalg.norm(grad_noise_[bi]) + 1e-10)
            noise[bi,:,:] += weight * grad_noise_[bi]
        if random_type == 'Gaussian-adversarial':
            grad_noise_[bi] /= (np.linalg.norm(grad_noise_[bi]) + 1e-10)
            noise[bi,:,:] += weight * grad_noise_[bi]
        if random_type == 'Bernoulli-adversarial':
            noise_flat = np.reshape(noise[bi], [-1])
            grad_noise_flat = np.reshape(grad_noise_[bi], [-1])#(seq_length * embedding_dim)
            grad_flat_abs = np.fabs(grad_noise_flat)
            sorted_id = np.argsort(grad_flat_abs)
            count_change = 0
            for i in range(len(grad_noise_flat)):
                id_ = sorted_id[i]
                if count_change > number_change:
                    break
                if noise_flat[id_] == 0 and grad_noise_flat[id_] > 0:
                    noise_flat[id_] = 1.0
                    count_change += 1
                elif noise_flat[id_] == 1 and grad_noise_flat[id_] < 0:
                    noise_flat[id_] = 0.0
                    count_change += 1
            noise[bi,:,:] = np.reshape(noise_flat, [seq_length, embedding_dim])
        if random_type == 'Bernoulli-word':
            x = list(x)
            x[bi] = x[bi] * np.random.choice(2, size=seq_length, p=[1-word_keep, word_keep])#change x by shallow copy
            #noise[bi,:,:] *= np.random.choice(2, size=(seq_length, 1), p=[1-word_keep, word_keep])
        if random_type == 'Bernoulli-semantic':
            noise[bi,:,:] *= np.random.choice(2, size=(embedding_dim), p=[1-word_keep, word_keep])
        if random_type == 'Bernoulli-idf':
            pass
        if random_type == 'Replace':
            positions = np.random.choice(2, size=seq_length, p=[1-word_keep, word_keep])
            positions = np.argwhere(positions == 0)
            for pi in positions:
                #print(x[bi])
                #print(int(x[bi][pi]))
                
                if int(x[bi][pi]) < len(replace_map) and replace_map[int(x[bi][pi])] != None:
                    word_choices, probs = replace_map[int(x[bi][pi])]
                    choose_wid = np.random.choice(word_choices, p=probs)
                    x[bi][pi] = choose_wid
                else:
                    x[bi][pi] = 0
            #x[bi] = np.random.choice(2, size=(seq_length), p=[1-word_keep, word_keep])#change x by shallow copy
    return noise
