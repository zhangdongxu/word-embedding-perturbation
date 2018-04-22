# On-going construction
# NON-debugged code

import argparse
import sys, os, socket
import numpy as np

from util.Vocab import Vocab
from util.read_data import read_relatedness_dataset, read_embedding

from model import *
from evaluator import *
from data_generate import * 

FLAGS = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.WARN)

def load_glove_word_embeddings(vocab):
    emb_dir = 'data/glove/'
    emb_prefix = emb_dir + 'glove.840B'
    emb_vocab, emb_vecs = read_embedding(emb_prefix + '.vocab', emb_prefix + '.300d.npy')
    emb_dim = emb_vecs.shape[1]

    # use only vectors in vocabulary (not necessary, but gives faster training)

    num_unk = 0
    vecs = np.zeros([vocab.size, emb_dim])

    UNK = np.random.uniform(low=-0.05, high=0.05, size=emb_dim)

    for i in range(0, vocab.size):
        w = vocab.token(i)
        if emb_vocab.contains(w):
            vecs[i] = emb_vecs[emb_vocab.index(w)]
        else:
            vecs[i] = emb_vecs[emb_vocab.index('unk')] #UNK --:uniform(-0.05, 0.05)
            num_unk = num_unk + 1
    print('unk count = %d' % num_unk)
    return vecs

def train(train_dataset, dev_dataset, test_dataset, vecs, iter_num = 100000):
    print '----MODEL TRAINING----'
    print 'Start building models'
    model_name = SentencePairEncoderMPSSN
    evaluator_name = SentencePairEvaluator
    data_generator_name = DataGeneratePointWise
    model = model_name(vecs, dim=FLAGS.dim, seq_length=FLAGS.max_length, regularization = FLAGS.reg, num_filters=[FLAGS.filterA,FLAGS.filterB])
    #check_op = tf.add_check_numerics_ops()
    if FLAGS.update_tech=='adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr)
    else:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.lr)
        
    saver = tf.train.Saver()
    #train = optimizer.minimize(model.loss)
    tvars = tf.trainable_variables() 
    grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), 10)
    train = optimizer.apply_gradients(zip(grads, tvars), global_step=model.global_step)
    grad_noise_q, grad_noise_a = tf.gradients(model.loss, [model.noise_q, model.noise_a])

    print('Start training')
    tf.set_random_seed(1234)
    np.random.seed(1234)
    best_dev_map, best_dev_mrr = 0, 0
    best_test_map, best_test_mrr = 0, 0
    best_model = None
    best_iter = 0
    not_improving = 0
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        data_generator = data_generator_name(model, sess, train_dataset, FLAGS.batch_size, FLAGS.max_length,  keep_prob=FLAGS.keep_prob)
        dev_evaluator = evaluator_name(model, sess, dev_dataset)
        test_evaluator = evaluator_name(model, sess, test_dataset)
        if FLAGS.load_model:
            saver.restore(sess, FLAGS.load_model)
        
        for iter in range(iter_num):
            feed_dict = data_generator.next(grad_noise_q, grad_noise_a, random_type=FLAGS.random_type, word_keep=FLAGS.word_keep, weight=FLAGS.noise_weight)
            _, loss = sess.run([train, model.loss], feed_dict=feed_dict)
             
            if iter%10 == 0: # TODO: change 2 to 50
                print('%d iter, loss = %.5f' %(iter, loss))
                sys.stdout.flush()
            if iter%FLAGS.eval_freq == 0:
                dev_map, dev_mrr = dev_evaluator.evaluate()
                test_map, test_mrr = test_evaluator.evaluate()
                if dev_map > best_dev_map:
                    not_improving = 0
                    best_dev_map, best_dev_mrr = dev_map, dev_mrr
                    best_test_map, best_test_mrr = test_map, test_mrr
                    best_iter = iter
                    print('New best valid MAP!')
                    saver.save(sess, FLAGS.save_path + '/model.tf')
                    print("Model saved")
                        
                else:
                    not_improving += 1
                    if not_improving > 3:
                        break
                print('%d iter, dev: MAP %.3f  MRR %.3f' %(iter, dev_map, dev_mrr))
                print('%d iter, test:  MAP %.3f  MRR %.3f' %(iter, test_map, test_mrr))
                print('Best at iter %d, valid %.3f, test %.3f\n' %(best_iter, best_dev_map, best_test_map))
                if not_improving > 3:
                    break
        print('\n\nTraining finished!')
        print('***************')
        print('Best at iter %d' %best_iter)
        print('Performance dev: MAP %.3f   MRR %.3f <==' %(best_dev_map, best_dev_mrr))
        print('Performance test: MAP %.3f   MRR %.3f' %(best_test_map, best_test_mrr))

def main(argv):
    if FLAGS.dataset != 'TrecQA' and FLAGS.dataset != 'WikiQA':
        print('Error dataset!')
        sys.exit()
    if FLAGS.save_path!='':
        FLAGS.save_path = FLAGS.save_path
        if not os.path.isdir(FLAGS.save_path):
            os.makedirs(FLAGS.save_path)
    else:
        save_path = [FLAGS.dataset]
        if FLAGS.dataset == 'TrecQA': 
            save_path.append(FLAGS.version)
        save_path.extend([FLAGS.model, FLAGS.random_type, str(FLAGS.word_keep), str(FLAGS.noise_weight), str(FLAGS.lr), str(FLAGS.time)])
        FLAGS.save_path = 'save_models/' + '_'.join(save_path)
        if not os.path.isdir(FLAGS.save_path):
            os.makedirs(FLAGS.save_path)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_file = open(FLAGS.save_path + '/log', 'w')
    sys.stdout = log_file
    sys.stderr = log_file
    
    print '----CONFIGURATION----'
    print 'hostname=%s' %socket.gethostname()
    try:
        print 'CUDA_VISIBLE_DEVICES=%s' %os.environ["CUDA_VISIBLE_DEVICES"]
    except:
        print 'Warning: CUDA_VISIBLE_DEVICES was not specified'
    print 'dataset=%s' %FLAGS.dataset
    print 'version=%s' %FLAGS.version
    print 'model=%s' %FLAGS.model
    print 'random_type=%s' %FLAGS.random_type
    print 'word_keep=%f' %FLAGS.word_keep
    print 'noise_weight=%f' %FLAGS.noise_weight
    print 'time=%d' %FLAGS.time

    print 'dim=%d' %FLAGS.dim
    print 'filterA=%d' %FLAGS.filterA
    print 'filterB=%d' %FLAGS.filterB
    print 'lr=%f' %FLAGS.lr
    print 'reg=%f' %FLAGS.reg
    print 'keep_prob=%f' %FLAGS.keep_prob
    print 'update_tech=%s' %FLAGS.update_tech
    print 'batch_size=%d' %FLAGS.batch_size
    print 'max_length=%d' %FLAGS.max_length
    print 'eval_freq=%d' %FLAGS.eval_freq
    print 'load_model=%s' %FLAGS.load_model
    print 'save_path=%s' %FLAGS.save_path
    print '**************\n\n'
    sys.stdout.flush()

    # directory containing dataset files
    data_dir = 'data/' + FLAGS.dataset + '/'

    # load vocab
    vocab = Vocab(data_dir + 'vocab.txt')

    # load embeddings
    print('loading glove word embeddings')
    vecs = load_glove_word_embeddings(vocab)

    # load datasets
    print('loading datasets' + FLAGS.dataset)
    if FLAGS.dataset == 'TrecQA':
        train_dir = data_dir + 'train-all/'
        dev_dir = data_dir + FLAGS.version + '-dev/'
        test_dir = data_dir + FLAGS.version + '-test/'
    elif FLAGS.dataset == 'WikiQA':
        train_dir = data_dir + 'train/'
        dev_dir = data_dir + 'dev/'
        test_dir = data_dir + 'test/'

    train_dataset = read_relatedness_dataset(train_dir, vocab, debug=False) #TODO: change debug to false # This is a dict
    dev_dataset = read_relatedness_dataset(dev_dir, vocab, debug=False)
    test_dataset = read_relatedness_dataset(test_dir, vocab, debug=False)
    print('train_dir: %s, num train = %d' % (train_dir, train_dataset['size']))
    print('dev_dir: %s, num dev = %d' % (dev_dir, dev_dataset['size']))
    print('test_dir: %s, num test = %d' % (test_dir, test_dataset['size']))

    train(train_dataset, dev_dataset, test_dataset, vecs)
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    log_file.close()

if __name__ == '__main__':
    tf.app.flags.DEFINE_string('dataset', 'TrecQA', 'dataset, can be TrecQA or WikiQA')
    tf.app.flags.DEFINE_string('version', 'clean', 'the version of TrecQA dataset, can be raw and clean')
    tf.app.flags.DEFINE_string('model', 'mpssn-pointwise', 'the version of model to be used')

    tf.app.flags.DEFINE_integer('dim', 150, 'dimension of hidden layers')
    tf.app.flags.DEFINE_integer('filterA', 300, 'number of filter A')
    tf.app.flags.DEFINE_integer('filterB', 20, 'number of filter B')
    tf.app.flags.DEFINE_float('lr', 0.0001, 'learning rate')
    tf.app.flags.DEFINE_float('reg', 0.0001, 'regularization weight')
    tf.app.flags.DEFINE_float('keep_prob', 0.5, 'keep probability of dropout during training')
    tf.app.flags.DEFINE_float('word_keep', 1.0, 'keep probability of dropout during training')
    tf.app.flags.DEFINE_float('noise_weight', 0.0, 'keep probability of dropout during training')
    tf.app.flags.DEFINE_string('random_type', 'None', 'gradient descent technique')
    tf.app.flags.DEFINE_integer('time', 1, 'dimension of hidden layers')
    tf.app.flags.DEFINE_string('update_tech', 'adam', 'gradient descent technique')
    tf.app.flags.DEFINE_integer('batch_size', 64, 'mini-batch size') #TODO: change batch size to 64
    tf.app.flags.DEFINE_integer('max_length', 48, 'max sentence length')
    tf.app.flags.DEFINE_integer('eval_freq', 1000, 'evaluate every x batches')
    tf.app.flags.DEFINE_string('load_model', '', 'specify a path to load a model')
    tf.app.flags.DEFINE_string('save_path', '', 'specify a path to save the best model')
    
    tf.app.run()
