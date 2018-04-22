import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string, data_type):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    if data_type[:4] == 'stsa':
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
        string = re.sub(r"\s{2,}", " ", string)
    else:
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower() if data_type[:4]!='TREC' else string.strip()


def count_label_num(data_file):
    max_class_id = max([int(line[0]) for line in open(data_file)])
    return max_class_id + 1

def load_data_and_labels(data_file, class_num):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    if data_file == '':
        return None, None
    # Load data from files
    examples = list(open(data_file, "r").readlines())
    examples = [s.strip() for s in examples]
    # Split by words
    x_text = [s[2:] for s in examples]
    x_text = [clean_str(sent, data_file) for sent in x_text]
    # Generate labels
    labels = []
    for s in examples:
        label_id = int(s[0])
        labels.append([1 if i==label_id else 0 for i in range(class_num)])
    y = np.array(labels)
    return [x_text, y]

def add_pad(x, filter_size):
    if x == None:
        return x
    pad = filter_size - 1
    new_x = np.zeros((x.shape[0], x.shape[1] + 2 * pad))
    new_x[:, pad:-pad] = x
    return new_x
    
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

def add_noise(sess, model, grad_noise, x, y, embedding_dim, random_type=None, word_keep=1.0, weight=0.0, replace_map = None):
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
            model.input_y: y,
            model.noise: noise,
            model.dropout_keep_prob: 1.0
        }
        grad_noise_ = sess.run(grad_noise, feed_dict=feed_dict)
    elif random_type == 'Gaussian-adversarial':
        noise = np.random.normal(1, weight, [batch_size, seq_length, embedding_dim])
        feed_dict = {
            model.input_x: x,
            model.input_y: y,
            model.noise: noise,
            model.dropout_keep_prob: 1.0
        }
        grad_noise_ = sess.run(grad_noise, feed_dict=feed_dict)
    elif random_type == 'Bernoulli-adversarial':
        noise = np.random.choice(2,size=(batch_size, seq_length, embedding_dim), p=[1-word_keep, word_keep])
        feed_dict = {
            model.input_x: x,
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
