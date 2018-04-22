#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import math
import os
import sys
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

# Data loading params
#tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_float("fold", 8, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_file", "./data/TRAIN_FILE.TXT", "Data source for the positive data.")
tf.flags.DEFINE_string("dev_file", "", "Data source for the positive data.")
tf.flags.DEFINE_string("test_file", "./data/TEST_FILE_FULL.TXT", "Data source for the positive data.")
tf.flags.DEFINE_string("word2vec_file", "./data/GoogleNews-vectors-negative300.bin", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("dist_dim", 10, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("word_keep", 1.0, "Word embedding dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("noise_weight", 0.0, "Word embedding dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
#tf.flags.DEFINE_float("lr", 1e-3, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_string("random_type", None, "Perturbation type")#Gaussian, Adversarial, Bernoulli, Bernoulli-adversarial, Bernoulli-word, Bernoulli-semantic 
tf.flags.DEFINE_integer("topk", 5, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("time", 1, "Batch Size (default: 64)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 25, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 2, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("train_emb", True, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

eval_filename = '.'.join([str(FLAGS.dist_dim), str(FLAGS.filter_sizes), str(FLAGS.num_filters), \
                         str(FLAGS.word_keep), str(FLAGS.noise_weight), str(FLAGS.random_type), str(FLAGS.time)])

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
rel2id, id2rel = data_helpers.count_label_num(FLAGS.train_file)
num_classes = len(rel2id)
x_train_text, train_e1_dist, train_e2_dist, y_train = data_helpers.load_data_and_labels(FLAGS.train_file, rel2id)
x_dev_text, dev_e1_dist, dev_e2_dist, y_dev = data_helpers.load_data_and_labels(FLAGS.dev_file, rel2id)
x_test_text, test_e1_dist, test_e2_dist, y_test = data_helpers.load_data_and_labels(FLAGS.test_file, rel2id)

# Build vocabulary
max_document_length = max([len(x.split()) for x in x_train_text])
mean_document_length = sum([len(x.split()) for x in x_train_text]) / float(len(x_train_text))
print('max_document_length=' + str(max_document_length))
print('average_document_length=' + str(mean_document_length))
#max_document_length = 50
count_longer = 0
for x in x_train_text:
    if len(x.split()) > max_document_length:
        count_longer += 1
print('%f percents of sentences longer than %d' %(count_longer * 100 / float(len(x_train_text)), max_document_length))
sys.stdout.flush()
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
if x_dev_text != None and x_test_text != None:
    vocab_processor = vocab_processor.fit(x_train_text + x_dev_text + x_test_text)
elif x_test_text != None:
    vocab_processor = vocab_processor.fit(x_train_text + x_test_text)
else:
    vocab_processor = vocab_processor.fit(x_train_text)
vocab_dict = vocab_processor.vocabulary_._mapping
x_train = np.array(list(vocab_processor.transform(x_train_text)))
x_dev = np.array(list(vocab_processor.transform(x_dev_text))) if x_dev_text != None else None
x_test = np.array(list(vocab_processor.transform(x_test_text))) if x_test_text != None else None

max_dist = 0
min_dist = 0
for batch in train_e1_dist:
    for d in batch:
        if d > max_dist:
            max_dist = d
        if d < min_dist:
            min_dist = d
for batch in train_e2_dist:
    for d in batch:
        if d > max_dist:
            max_dist = d
        if d < min_dist:
            min_dist = d
dist_vocab_size = max_dist - min_dist + 2
shift_dist = 1 - min_dist
print('dist_vocab_size=' + str(dist_vocab_size))
def list_add(e_dist, shift_dist):
    if e_dist == None:
        return None
    new_dist = []
    for batch in e_dist:
        new_dist.append([])
        for d in batch:
            
            new_d = d + shift_dist
            if new_d < 1:
                new_d = 1
            if new_d >= dist_vocab_size:
                new_d = dist_vocab_size - 1
            new_dist[-1].append(new_d)
    return new_dist

train_e1_dist = list_add(train_e1_dist, shift_dist)
train_e2_dist = list_add(train_e2_dist, shift_dist)
dev_e1_dist = list_add(dev_e1_dist, shift_dist)
dev_e2_dist = list_add(dev_e2_dist, shift_dist)
test_e1_dist = list_add(test_e1_dist, shift_dist)
test_e2_dist = list_add(test_e2_dist, shift_dist)

train_e1_dist = data_helpers.padding(train_e1_dist, max_document_length)
train_e2_dist = data_helpers.padding(train_e2_dist, max_document_length)
dev_e1_dist = data_helpers.padding(dev_e1_dist, max_document_length)
dev_e2_dist = data_helpers.padding(dev_e2_dist, max_document_length)
test_e1_dist = data_helpers.padding(test_e1_dist, max_document_length)
test_e2_dist = data_helpers.padding(test_e2_dist, max_document_length)

x_train, train_e1_dist, train_e2_dist = data_helpers.add_pad(x_train, train_e1_dist, train_e2_dist, 5)
x_dev, dev_e1_dist, dev_e2_dist = data_helpers.add_pad(x_dev, dev_e1_dist, dev_e2_dist, 5)
x_test, test_e1_dist, test_e2_dist = data_helpers.add_pad(x_test, test_e1_dist, test_e2_dist, 5)


def load_bin_vec_replace(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = np.random.uniform(-0.25, 0.25, [len(vocab), 300])
    with open(fname, "rb") as f:
        header = f.readline()
        origin_vocab_size = len(vocab)
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        
        wordvec_all = np.zeros((vocab_size, 300))
        vocab_all = [None for _ in range(vocab_size)]#word list of wordvec_all
        wid = 0
        wordvec_exist_list = []#words in vocab whose word vec exist.
        
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            vec = np.fromstring(f.read(binary_len), dtype='float32')
            if '_' not in word and '.com' not in word and 'http' not in word \
               and '@' not in word and '/' not in word and '#' not in word \
               and word.lower() == word:
                vocab_all[wid] = word
                wordvec_all[wid] = vec
                wid += 1
            if word in vocab:
                wordvec_exist_list.append(word)
                word_vecs[vocab[word]] = vec
        wordvec_all = wordvec_all[:wid]
        origin_word_vecs = word_vecs
        word_vecs = list(word_vecs)
        wordvec_all_l2norm = np.sqrt(np.sum(wordvec_all ** 2,axis=1)) + 1e-12
        replace_map = [None for _ in range(len(vocab))]
        topk=FLAGS.topk
        batch_size = 2000
        for i in range(int(math.ceil(len(wordvec_exist_list)/float(batch_size)))):
            batch = wordvec_exist_list[i*batch_size:(i+1)*batch_size]
            print(i, int(math.ceil(len(wordvec_exist_list)/float(batch_size))))
            print(batch)
            sys.stdout.flush()

            #TODO: Speed up
            batch_ = batch
            batch = [vocab[word] for word in batch]
            sims = np.dot(origin_word_vecs[batch], wordvec_all.T)
            sims /= (np.sqrt(np.sum(origin_word_vecs[batch] **2, axis=1)).reshape([len(batch), 1]) + 1e-12)
            sims /= wordvec_all_l2norm
            sims_ = np.copy(sims)
            max_wids = []
            for k in range(topk+1):
                max_wids.append(np.argmax(sims_,axis=1))
                sims_[range(len(batch)), max_wids[-1]] = -100
            max_wids = np.array(max_wids).T[:,1:topk+1]
            #max_wids = np.argsort(sims, axis = 1)[:,-topk:][:,::-1]
            #print(max_wids)
            sys.stdout.flush()

            #TODO: Speed up
            words = {}
            for wid in list(np.reshape(max_wids, [-1])):
                word = vocab_all[wid]
                if word not in words:
                    words[word] = wid
            for word, wid in words.iteritems():
                if word not in vocab:
                    vec = wordvec_all[wid]
                    vocab[word] = len(vocab)
                    word_vecs.append(vec)
            #print('Done')

            sys.stdout.flush()
            sims = sims[np.array(range(max_wids.shape[0]*max_wids.shape[1]))/topk, np.reshape(max_wids, [-1])].reshape(max_wids.shape)
            probs = np.exp(sims)
            probs = sims / np.sum(sims, axis=1).reshape([len(batch), 1])
            #print('Done2')
            sys.stdout.flush()
            for j in range(len(batch)):
                word_list = []
                word_list_ = []
                for wid in max_wids[j]:
                    word_list.append(vocab[vocab_all[wid]])
                    word_list_.append(vocab_all[wid])
                print(batch_[j], word_list_, sims[j], probs[j])
                sys.stdout.flush()
                replace_map[batch[j]] = [word_list, list(probs[j])]
        replace_map[0] = None
        word_vecs = np.array(word_vecs)
    print('Initialized %d/%d word embeddings, extend %d extra embeddings for replacement' %(len(wordvec_exist_list), origin_vocab_size, len(vocab) - origin_vocab_size))
    return word_vecs, replace_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = np.random.uniform(-0.25, 0.25, [len(vocab), 300])
    with open(fname, "rb") as f:
        count = 0
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                count += 1
                word_vecs[vocab[word]] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    print('Initialized %d/%d word embeddings' %(count, len(vocab)))
    return word_vecs

if FLAGS.random_type != 'Replace': 
    word_vecs = load_bin_vec(FLAGS.word2vec_file, vocab_dict)
    replace_map = None
else:
    word_vecs, replace_map = load_bin_vec_replace(FLAGS.word2vec_file, vocab_dict)
    #word_vecs = load_bin_vec(FLAGS.word2vec_file, vocab_dict)
    #replace_map = None

wid2word = {}
for w, wid in vocab_dict.iteritems():
    wid2word[wid] = w
# Randomly shuffle data
np.random.seed(3435)
shuffle_indices = np.random.permutation(np.arange(len(x_train)))
x_shuffled = x_train[shuffle_indices].copy()
e1_dist_shuffled = train_e1_dist[shuffle_indices].copy()
e2_dist_shuffled = train_e2_dist[shuffle_indices].copy()
y_shuffled = y_train[shuffle_indices].copy()



if x_dev == None and x_test != None:
    valid_size = len(x_shuffled)/FLAGS.fold
    x_trains, x_devs, x_tests = [], [], []
    train_e1_dists, dev_e1_dists, test_e1_dists = [], [], []
    train_e2_dists, dev_e2_dists, test_e2_dists = [], [], []
    y_trains, y_devs, y_tests = [], [], []
    x_shuffled = list(x_shuffled)
    e1_dist_shuffled = list(e1_dist_shuffled)
    e2_dist_shuffled = list(e2_dist_shuffled)
    y_shuffled = list(y_shuffled)
    for i in range(FLAGS.fold):
        range_test = (i*valid_size, (i+1)*valid_size)

        x_devs.append(np.array(x_shuffled[range_test[0]:range_test[1]]))
        dev_e1_dists.append(np.array(e1_dist_shuffled[range_test[0]:range_test[1]]))
        dev_e2_dists.append(np.array(e2_dist_shuffled[range_test[0]:range_test[1]]))
        y_devs.append(np.array(y_shuffled[range_test[0]:range_test[1]]))

        x_train = x_shuffled[:range_test[0]] + x_shuffled[range_test[1]:]
        train_e1_dist = e1_dist_shuffled[:range_test[0]] + e1_dist_shuffled[range_test[1]:]
        train_e2_dist = e2_dist_shuffled[:range_test[0]] + e2_dist_shuffled[range_test[1]:]
        y_train = y_shuffled[:range_test[0]] + y_shuffled[range_test[1]:]

        x_trains.append(np.array(x_train))
        train_e1_dists.append(np.array(train_e1_dist))
        train_e2_dists.append(np.array(train_e2_dist))
        y_trains.append(np.array(y_train))

        x_tests.append(np.array(x_test))
        test_e1_dists.append(np.array(test_e1_dist))
        test_e2_dists.append(np.array(test_e2_dist))
        y_tests.append(np.array(y_test))
    
elif x_dev != None and x_test != None:
    x_trains, x_devs, x_tests = [], [], []
    train_e1_dists, dev_e1_dists, test_e1_dists = [], [], []
    train_e2_dists, dev_e2_dists, test_e2_dists = [], [], []
    y_trains, y_devs, y_tests = [], [], []

    x_trains.append(np.array(x_shuffled))
    train_e1_dists.append(np.array(e1_dist_shuffled))
    train_e2_dists.append(np.array(e2_dist_shuffled))
    y_trains.append(np.array(y_shuffled))

    x_devs.append(np.array(x_dev))
    dev_e1_dists.append(np.array(dev_e1_dist))
    dev_e2_dists.append(np.array(dev_e2_dist))
    y_devs.append(np.array(y_dev))
    
    x_tests.append(np.array(x_test))
    test_e1_dists.append(np.array(test_e1_dist))
    test_e2_dists.append(np.array(test_e2_dist))
    y_tests.append(np.array(y_test))

#dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
#x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
#y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
#print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_trains[0].shape[1],
            num_classes=num_classes,
            vocab_size=len(word_vecs),
            embedding_size=FLAGS.embedding_dim,
            dist_vocab_size=dist_vocab_size,
            dist_size=FLAGS.dist_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            word_vecs = word_vecs,
            train_emb=FLAGS.train_emb)
        del word_vecs
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdadeltaOptimizer(1.0,rho=0.95,epsilon=1e-06)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cnn.loss, tvars), 3)
        grads_and_vars = zip(grads, tvars)
        #grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        grad_noise = tf.gradients(cnn.loss, [cnn.noise])[0]

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        def train_step(x_batch, e1_dist, e2_dist, noise_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.e1_dist: e1_dist,
              cnn.e2_dist: e2_dist,
              cnn.input_y: y_batch,
              cnn.noise: noise_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss],
                feed_dict)
            train_summary_writer.add_summary(summaries, step)
            return loss

        def dev_step(x_batch, e1_dist, e2_dist, noise_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.e1_dist: e1_dist,
              cnn.e2_dist: e2_dist,
              cnn.input_y: y_batch,
              cnn.noise: noise_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, predictions = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.predictions],
                feed_dict)
            fout = open('predict.'+eval_filename, 'w')
            for i in range(len(x_batch)):
                fout.write(str(i+8000) + '\t' + id2rel[predictions[i]] + '\n')
            fout.close()
            fout = open('answer.' + eval_filename, 'w')
            for i in range(len(x_batch)):
                fout.write(str(i+8000) + '\t' + id2rel[int(np.argmax(y_batch[i]))] + '\n')
            fout.close()
            os.system('perl data/semeval2010_task8_scorer-v1.2.pl %s %s > %s ' %('predict.'+eval_filename, 'answer.'+eval_filename, 'result.'+eval_filename))
            f1 = open('result.'+eval_filename).read().split('\n')[-2]
            f1 = f1.split()[-2][:-1]
            f1 = float(f1)/100
            if writer:
                writer.add_summary(summaries, step)
            return loss, f1

        # Initialize all variables
        # Generate batches
        devf1s = np.zeros((len(x_trains)))
        testf1s = np.zeros((len(x_trains)))
        best_steps = []
        for fi in range(len(x_trains)):
            sess.run(tf.global_variables_initializer())
            batches = data_helpers.batch_iter(
                list(zip(x_trains[fi], train_e1_dists[fi], train_e2_dists[fi],  y_trains[fi])), FLAGS.batch_size, FLAGS.embedding_dim, FLAGS.num_epochs)
            best_dev_f1 = 0
            best_test_f1 = 0
            overfit_num = 0
            best_steps.append(0)
            # Training loop. For each batch...
            evaluate_every = len(x_trains[fi])/FLAGS.batch_size
            for batch in batches:
                x_batch, e1_dist, e2_dist, y_batch = zip(*batch)
                #for x_ in x_batch:
                    #print(' '.join([wid2word[wid] for wid in x_]))
                noise = data_helpers.add_noise(sess, cnn, grad_noise,
                    x_batch, e1_dist, e2_dist, y_batch, FLAGS.embedding_dim, random_type=FLAGS.random_type, 
                    word_keep=FLAGS.word_keep, weight=FLAGS.noise_weight, replace_map=replace_map)
                #for x_ in x_batch:
                    #print(' '.join([wid2word[wid] for wid in x_]))
                train_loss = train_step(x_batch, e1_dist, e2_dist, noise, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                print("step {}, loss {:g} ".format(current_step, train_loss))
                sys.stdout.flush()
                #if current_step % FLAGS.evaluate_every == 0:
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    noise = np.ones([x_devs[fi].shape[0], x_devs[fi].shape[1], FLAGS.embedding_dim])
                    dev_loss, dev_f1 = dev_step(x_devs[fi], dev_e1_dists[fi], dev_e2_dists[fi], noise, y_devs[fi], writer=dev_summary_writer)
                    noise = np.ones([x_tests[fi].shape[0], x_tests[fi].shape[1], FLAGS.embedding_dim])
                    test_loss, test_f1 = dev_step(x_tests[fi], test_e1_dists[fi], test_e2_dists[fi], noise, y_tests[fi], writer=dev_summary_writer)
                    print("step {}, dev f1 {:g}, test f1 {:g}".format(current_step, dev_f1, test_f1))
                    if dev_f1 < best_dev_f1:
                        overfit_num += 1
                    else:
                        overfit_num = 0
                        best_dev_f1 = dev_f1
                        best_test_f1 = test_f1
                        devf1s[fi] = best_dev_f1
                        testf1s[fi] = best_test_f1
                        best_steps[-1] = current_step
            print("")
        print('Overall devf1 %f testf1 %f' %(devf1s.mean(), testf1s.mean()))
        
        # run the whole training set with best hyper-parameters:
        sess.run(tf.global_variables_initializer())
        batches = data_helpers.batch_iter(
                list(zip(x_shuffled, e1_dist_shuffled, e2_dist_shuffled,  y_shuffled)), FLAGS.batch_size, FLAGS.embedding_dim, FLAGS.num_epochs)
        step_num = np.mean(best_steps)
        print('Average best step number: %f' %step_num)
        for batch in batches:
            x_batch, e1_dist, e2_dist, y_batch = zip(*batch)
            noise = data_helpers.add_noise(sess, cnn, grad_noise,
                    x_batch, e1_dist, e2_dist, y_batch, FLAGS.embedding_dim, random_type=FLAGS.random_type,
                    word_keep=FLAGS.word_keep, weight=FLAGS.noise_weight, replace_map=replace_map)
            train_loss = train_step(x_batch, e1_dist, e2_dist, noise, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            print("step {}, loss {:g} ".format(current_step, train_loss))
            sys.stdout.flush()
            if current_step > step_num:
                noise = np.ones([x_tests[fi].shape[0], x_tests[fi].shape[1], FLAGS.embedding_dim])
                test_loss, test_f1 = dev_step(x_tests[0], test_e1_dists[0], test_e2_dists[0], noise, y_tests[0], writer=dev_summary_writer)
                print('Final testf1 %f' %test_f1)
                break
