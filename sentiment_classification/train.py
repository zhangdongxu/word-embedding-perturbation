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
tf.flags.DEFINE_float("fold", 10, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_file", "./data/rt-polarity", "Data source for the positive data.")
tf.flags.DEFINE_string("dev_file", "", "Data source for the positive data.")
tf.flags.DEFINE_string("test_file", "", "Data source for the positive data.")
tf.flags.DEFINE_string("word2vec_file", "./data/GoogleNews-vectors-negative300.bin", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 100, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("word_keep", 1.0, "Word embedding dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("noise_weight", 0.0, "Word embedding dropout keep probability (default: 1.0)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("lr", 1e-3, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_string("random_type", None, "Perturbation type")#Gaussian, Adversarial, Bernoulli, Bernoulli-adversarial, Bernoulli-word, Bernoulli-semantic 
tf.flags.DEFINE_integer("topk", 5, "Batch Size (default: 64)")

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


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
num_classes = data_helpers.count_label_num(FLAGS.train_file)
x_train_text, y_train = data_helpers.load_data_and_labels(FLAGS.train_file, num_classes)
x_dev_text, y_dev = data_helpers.load_data_and_labels(FLAGS.dev_file, num_classes)
x_test_text, y_test = data_helpers.load_data_and_labels(FLAGS.test_file, num_classes)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_train_text])
mean_document_length = sum([len(x.split(" ")) for x in x_train_text]) / float(len(x_train_text))
print('max_document_length=' + str(max_document_length))
print('average_document_length=' + str(mean_document_length))
#max_document_length = 50
count_longer = 0
for x in x_train_text:
    if len(x.split(" ")) > max_document_length:
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

x_train = data_helpers.add_pad(x_train, 5)
x_dev = data_helpers.add_pad(x_dev, 5)
x_test = data_helpers.add_pad(x_test, 5)



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
x_shuffled = x_train[shuffle_indices]
y_shuffled = y_train[shuffle_indices]



if x_dev == None and x_test == None:
    test_size = len(x_shuffled)/FLAGS.fold
    x_trains, x_devs, x_tests = [], [], []
    y_trains, y_devs, y_tests = [], [], []
    x_shuffled = list(x_shuffled)
    y_shuffled = list(y_shuffled)
    valid_size = (len(x_shuffled) - test_size) / FLAGS.fold
    for i in range(FLAGS.fold):
        range_test = (i*test_size, (i+1)*test_size)
        x_tests.append(np.array(x_shuffled[range_test[0]:range_test[1]]))
        y_tests.append(np.array(y_shuffled[range_test[0]:range_test[1]]))
        x_train = x_shuffled[:range_test[0]] + x_shuffled[range_test[1]:]
        y_train = y_shuffled[:range_test[0]] + y_shuffled[range_test[1]:]
        x_devs.append(np.array(x_train[-valid_size:]))
        y_devs.append(np.array(y_train[-valid_size:]))
        x_trains.append(np.array(x_train[:-valid_size]))
        y_trains.append(np.array(y_train[:-valid_size]))
elif x_dev == None and x_test != None:
    valid_size = len(x_shuffled)/FLAGS.fold
    x_trains, x_devs, x_tests = [], [], []
    y_trains, y_devs, y_tests = [], [], []
    x_shuffled = list(x_shuffled)
    y_shuffled = list(y_shuffled)
    for i in range(FLAGS.fold):
        range_test = (i*valid_size, (i+1)*valid_size)
        x_devs.append(np.array(x_shuffled[range_test[0]:range_test[1]]))
        y_devs.append(np.array(y_shuffled[range_test[0]:range_test[1]]))
        x_train = x_shuffled[:range_test[0]] + x_shuffled[range_test[1]:]
        y_train = y_shuffled[:range_test[0]] + y_shuffled[range_test[1]:]
        x_trains.append(np.array(x_train))
        y_trains.append(np.array(y_train))
        x_tests.append(np.array(x_test))
        y_tests.append(np.array(y_test))
elif x_dev != None and x_test != None:
    x_trains, x_devs, x_tests = [], [], []
    y_trains, y_devs, y_tests = [], [], []
    x_trains.append(np.array(x_shuffled))
    y_trains.append(np.array(y_shuffled))
    x_devs.append(np.array(x_dev))
    y_devs.append(np.array(y_dev))
    x_tests.append(np.array(x_test))
    y_tests.append(np.array(y_test))

#dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
#x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
#y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x_train, y_train, x_shuffled, y_shuffled

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
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            word_vecs = word_vecs,
            train_emb=FLAGS.train_emb)
        del word_vecs
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        optimizer = tf.train.AdadeltaOptimizer(1.0,rho=0.95,epsilon=1e-06)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cnn.loss, tvars), 10)
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

        def train_step(x_batch, noise_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.noise: noise_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            train_summary_writer.add_summary(summaries, step)
            return loss, accuracy

        def dev_step(x_batch, noise_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.noise: noise_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            if writer:
                writer.add_summary(summaries, step)
            return loss, accuracy

        # Initialize all variables
        # Generate batches
        devaccs = np.zeros((len(x_trains)))
        testaccs = np.zeros((len(x_trains)))
        for fi in range(len(x_trains)):
            sess.run(tf.global_variables_initializer())
            batches = data_helpers.batch_iter(
                list(zip(x_trains[fi], y_trains[fi])), FLAGS.batch_size, FLAGS.embedding_dim, FLAGS.num_epochs)
            best_dev_acc = 0
            best_test_acc = 0
            overfit_num = 0
            # Training loop. For each batch...
            evaluate_every = len(x_trains[fi])/FLAGS.batch_size
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                #for x_ in x_batch:
                    #print(' '.join([wid2word[wid] for wid in x_]))
                noise = data_helpers.add_noise(sess, cnn, grad_noise,
                    x_batch, y_batch, FLAGS.embedding_dim, random_type=FLAGS.random_type, 
                    word_keep=FLAGS.word_keep, weight=FLAGS.noise_weight, replace_map=replace_map)
                #for x_ in x_batch:
                    #print(' '.join([wid2word[wid] for wid in x_]))
                train_loss, train_acc = train_step(x_batch, noise, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                print("step {}, loss {:g}, trainacc {:g}".format(current_step, train_loss, train_acc))
                sys.stdout.flush()
                #if current_step % FLAGS.evaluate_every == 0:
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    noise = np.ones([x_devs[fi].shape[0], x_devs[fi].shape[1], FLAGS.embedding_dim])
                    dev_loss, dev_acc = dev_step(x_devs[fi], noise, y_devs[fi], writer=dev_summary_writer)
                    noise = np.ones([x_tests[fi].shape[0], x_tests[fi].shape[1], FLAGS.embedding_dim])
                    test_loss, test_acc = dev_step(x_tests[fi], noise, y_tests[fi], writer=dev_summary_writer)
                    print("step {}, devacc {:g}, testacc {:g}".format(current_step, dev_acc, test_acc))
                    if dev_acc < best_dev_acc:
                        overfit_num += 1
                    else:
                        overfit_num = 0
                        best_dev_acc = dev_acc
                        best_test_acc = test_acc
                        devaccs[fi] = best_dev_acc
                        testaccs[fi] = best_test_acc
            print("")
        print('Overall devacc %f testacc %f' %(devaccs.mean(), testaccs.mean()))
