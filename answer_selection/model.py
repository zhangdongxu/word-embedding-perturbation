import tensorflow as tf
import numpy as np
from util.MPSSN_utils import *

def init_weight(shape, name):
  #return tf.Variable(tf.truncated_normal(shape, stddev=0.01), name=name)
  return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), trainable=True)

class SentencePairEncoder(object):
    def __init__(self, pretrained_embeddings, dim=100,
                 use_tanh=False, verbose=False, 
                 dropout_keep=1.0, seq_length=32, regularization=0.01,
                 num_filters=[300,20], filter_sizes=[1,2,3,100]):
        self._vocab_size = pretrained_embeddings.shape[0]
        self._embed_dim = pretrained_embeddings.shape[1]
        self._dim = dim
        self._non_linear = tf.nn.tanh if use_tanh else tf.nn.relu
        self._verbose = verbose
        self._dropout_keep = dropout_keep
        self._seq_length = seq_length
        self._regularization = regularization
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        self.input_questions = tf.placeholder(tf.int32, [None, seq_length], name='input_questions')
        self.input_question_lens = tf.placeholder(tf.int32, [None], name='input_question_lens')
        self.input_answers = tf.placeholder(tf.int32, [None, seq_length], name='input_answers')
        self.input_answer_lens = tf.placeholder(tf.int32, [None], name='input_answer_lens')
        self.keep_prob = tf.placeholder(tf.float32)
        self.noise_q = tf.placeholder(tf.float32, [None, seq_length, self._embed_dim], name="noise_q")
        self.noise_a = tf.placeholder(tf.float32, [None, seq_length, self._embed_dim], name="noise_a")
        self.word_embeddings = tf.get_variable(name='word_embeddings',
                                               initializer=tf.constant(pretrained_embeddings, dtype=tf.float32),
                                               trainable=False)

        # For MPSSN:
        self.num_filters = num_filters  # [num_filters_A, num_filters_B]
        self.filter_sizes = filter_sizes   # [1,2,3,100]
        self.filter_sizes[-1] = self._seq_length
        self.ngram = len(filter_sizes)-1

        self.init_extra()
        self.derive_loss()

    def init_extra(self):
        '''
        This function could be override in a child class 
        for extra placeholders or parameters.
        '''
        self.labels = tf.placeholder(tf.int32, [None, 2], name='labels')
        self.linear_matrix = tf.get_variable(name='linear_matrix',
                                               shape=[self._embed_dim * 2, self._dim],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        self.score_vector = tf.get_variable(name='score_vector',
                                               shape=[self._dim, 2],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
    
    def produce_feature(self, sentences, lens):
        '''
        This function could be overrided in a child class
        to generate qa pair features from word embedding layer.
        '''
        #(batchsize, seq_length, embed_size)
        embed_layer = tf.nn.embedding_lookup(self.word_embeddings, sentences)
        embed_layer_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * embed_layer
        features = tf.reduce_sum(embed_layer_mask, axis=1) / tf.expand_dims(tf.cast(lens, tf.float32), 1)
        
        #(batchsize, 1, embed_size, 1)
        '''
        q_features = tf.nn.pool(q_embed_layer,
                                      window_shape = [1, self._seq_length, 1, 1], 
                                      strides = [1, self._seq_length, 1, 1],
                                      pooling_type = 'AVG',
                                      padding = 'SAME',
                                      name = 'word_embedding_pooling')
        a_features = tf.nn.pool(a_embed_layer,
                                      window_shape = [1, self._seq_length, 1, 1], 
                                      strides = [1, self._seq_length, 1, 1],
                                      pooling_type = 'AVG',
                                      padding = 'SAME',
                                      name = 'word_embedding_pooling')
        '''
        return features
    
    def derive_loss(self):
        '''
        This function could be overided in a child class
        to derive score and loss.
        '''
        q_features = self.produce_feature(self.input_questions, self.input_question_lens) 
        a_features = self.produce_feature(self.input_answers, self.input_answer_lens) 
        pair_features = tf.concat([q_features, a_features], axis=1)
        pair_features = tf.nn.relu(tf.matmul(pair_features, self.linear_matrix))#(batchsize, last_dim)
        pair_features = tf.nn.dropout(pair_features, self.keep_prob)
        scores = tf.matmul(pair_features, self.score_vector)#(batchsize, 2)
        self.scores = scores[:, 1]#(batchsize,)
        losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.labels))
        #losses = - tf.cast(self.labels, tf.float32) * tf.log(probs + 1e-8) \
        #         - (1.0 - tf.cast(self.labels, tf.float32)) * tf.log(1.0 - probs + 1e-8)
        self.loss = tf.reduce_mean(losses)
        #l2_loss = tf.constant(l2_weight) * tf.nn.l2_loss()

class SentencePairEncoderMPSSN(SentencePairEncoder):
  def init_extra(self):
    '''
    This function could be override in a child class 
    for extra placeholders or parameters.
    '''
    self.num_classes = 2
    self.labels = tf.placeholder(tf.int32, [None, self.num_classes], name='labels')
    self.poolings = [tf.reduce_max, tf.reduce_min, tf.reduce_mean]

    self.W1 = [init_weight([self.filter_sizes[0], self._embed_dim, 1, self.num_filters[0]], "W1_0"),
               init_weight([self.filter_sizes[1], self._embed_dim, 1, self.num_filters[0]], "W1_1"),
               init_weight([self.filter_sizes[2], self._embed_dim, 1, self.num_filters[0]], "W1_2")]
    self.b1 = [tf.Variable(tf.constant(0.1, shape=[self.num_filters[0]]), "b1_0"),
               tf.Variable(tf.constant(0.1, shape=[self.num_filters[0]]), "b1_1"),
               tf.Variable(tf.constant(0.1, shape=[self.num_filters[0]]), "b1_2")]
    self.W2 = [init_weight([self.filter_sizes[0], self._embed_dim, 1, self.num_filters[1]], "W2_0"),
               init_weight([self.filter_sizes[1], self._embed_dim, 1, self.num_filters[1]], "W2_1"),
               init_weight([self.filter_sizes[2], self._embed_dim, 1, self.num_filters[1]], "W2_2")]
    self.b2 = [tf.Variable(tf.constant(0.1, shape=[self.num_filters[1], self._embed_dim]), "b2_0"),
               tf.Variable(tf.constant(0.1, shape=[self.num_filters[1], self._embed_dim]), "b2_1"),
               tf.Variable(tf.constant(0.1, shape=[self.num_filters[1], self._embed_dim]), "b2_2")]
    # items = (self.ngram + 1)*3
    # inputNum = 2*items*items/3+NumFilter*items*items/3+6*NumFilter+(2+NumFilter)*2*ngram*conceptFNum --PoinPercpt model!
    # self.h = 2*items*items/3 + self.num_filters[0]*items*items/3 + 6*self.num_filters[0] + (2+self.num_filters[0])*2*self.ngram*self.num_filters[1]
    self.h = 2*(3*self.num_filters[0]) + (self.num_filters[0]+2)*(3*(self.ngram + 1)*(self.ngram + 1)) + (self._embed_dim+2)*(2*self.num_filters[1]*3)
    print 'feature dimension: ' + str(self.h)
    self.Wh = init_weight([self.h, self._dim], 'Wh')
    #self.Wh = tf.Variable(tf.random_normal([self.h, self._dim], stddev=0.01), name='Wh')
    self.bh = tf.Variable(tf.constant(0.1, shape=[self._dim]), name="bh")

    self.Wo = init_weight([self._dim, self.num_classes], 'Wo')
    #self.Wo = tf.Variable(tf.random_normal([self._dim, self.num_classes], stddev=0.01), name='Wo')

  def produce_feature(self, sentences, lens):
    with tf.name_scope("embedding"):
      embed_layer = tf.nn.embedding_lookup(self.word_embeddings, sentences)
      
      self.sentences = sentences
      self.embed_layer = embed_layer
      embed_layer_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * embed_layer
      #embed_layer_mask = embed_layer
      self.embed_layer_mask = embed_layer_mask
    with tf.name_scope("reshape"):
      result = tf.reshape(embed_layer_mask, [-1, self._seq_length, self._embed_dim, 1])
    return result

  def derive_loss(self):
    #(batchsize, sentence_length, embed_dimmension, 1)
    q_features = self.produce_feature(self.input_questions, self.input_question_lens) 
    a_features = self.produce_feature(self.input_answers, self.input_answer_lens)
    q_features = q_features * tf.reshape(self.noise_q, [-1, self._seq_length, self._embed_dim, 1])
    a_features = a_features * tf.reshape(self.noise_a, [-1, self._seq_length, self._embed_dim, 1])
    # self.labels = tf.expand_dims(self.labels, 1)
    # bloack A
    sent1 = self.bulit_block_A(q_features)#pool_type_num, window_type_nums, (batch_size, 1, num_filter_A)
    sent2 = self.bulit_block_A(a_features)
    fea_h = []#3 * num_filter_A * (batchsize, 2) 
    with tf.name_scope("cal_dis_with_alg1"):
      for i in range(3):
        regM1 = tf.concat(sent1[i], 1)#(batch_size, window_type_nums, num_filter_A)
        regM2 = tf.concat(sent2[i], 1)
        for k in range(self.num_filters[0]):
          fea_h.append(comU2(regM1[:, :, k], regM2[:, :, k]))#(batchsize,2)
    fea_a = []#3 * 4 * 4 * (batchsize, num_filter_A + 2)
    with tf.name_scope("cal_dis_with_alg2_2-9"):
      for i in range(3):
        for j in range(len(self.filter_sizes)):
          for k in range(len(self.filter_sizes)):
            fea_a.append(comU1(sent1[i][j][:, 0, :], sent2[i][k][:, 0, :]))#(batchsize, num_filter_A + 2)
    # bloack B
    sent1 = self.bulid_block_B(q_features)#2 * 3 * 
    sent2 = self.bulid_block_B(a_features)
    fea_b = []
    with tf.name_scope("cal_dis_with_alg2_last"):
      for i in range(len(self.poolings)-1):
        for j in range(len(self.filter_sizes)-1):
          for k in range(self.num_filters[1]):
            fea_b.append(comU1(sent1[i][j][:, :, k], sent2[i][j][:, :, k]))
    # concate all features together
    fea = tf.concat(fea_h + fea_b + fea_a, 1)
    self.fea_h = fea_h
    self.fea_a = fea_a
    self.fea_b = fea_b
    self.fea = fea
    # FC layer
    with tf.name_scope("full_connect_layer"):
      # print fea.get_shape()
      # print self.Wh.get_shape()
      h = tf.nn.tanh(tf.matmul(fea, self.Wh) + self.bh)
      h = tf.nn.dropout(h, self.keep_prob)
      out = tf.matmul(h, self.Wo, name='out')
    # print out.get_shape()
    # Calc score for evaluation
    self.out = out
    softmax_result = tf.nn.softmax(logits=out)
    self.scores = softmax_result[:, 1]
    # Calc loss
    losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=self.labels),name='loses')
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.trainable_variables() if 'word_embeddings' not in w.name])
    self.l2_loss = l2_loss
     
    self.loss = tf.reduce_mean(losses) + self._regularization * l2_loss

  def per_dim_conv_layer(self, x, w, b, pooling):
    '''
    :param input: [batch_size, sentence_length, embed_size, 1]
    :param w: [ws, embedding_size, 1, num_filters]
    :param b: [num_filters, embedding_size]
    :param pooling:
    :return:
    '''
    # unpcak the input in the dim of embed_dim
    input_unstack = tf.unstack(x, axis=2)
    w_unstack = tf.unstack(w, axis=1)
    b_unstack = tf.unstack(b, axis=1)
    convs = []
    for i in range(x.get_shape()[2]):
      conv = tf.nn.relu(tf.nn.conv1d(input_unstack[i], w_unstack[i], stride=1, padding="VALID") + b_unstack[i])
      # [batch_size, sentence_length-ws+1, num_filters_A]
      convs.append(conv)
    conv = tf.stack(convs, axis=2)  # [batch_size, sentence_length-ws+1, embed_size, num_filters_A]
    pool = pooling(conv, axis=1)  # [batch_size, embed_size, num_filters_A]
    return pool

  def bulit_block_A(self, x):
    #bulid block A and cal the similarity according to algorithm 1
    out = []
    with tf.name_scope("bulid_block_A"):
      for pooling in self.poolings:
        pools = []
        with tf.name_scope("pool-ws-infinite"):
          pool = pooling(x, axis=1)
          pools.append(tf.reshape(pool, [-1, 1, self._embed_dim]))
        for i, ws in enumerate(self.filter_sizes[:-1]): # ws: window size
          #print x.get_shape(), self.W1[i].get_shape()
          with tf.name_scope("conv-ws%d" %ws):
            conv = tf.nn.conv2d(x, self.W1[i], strides=[1, 1, 1, 1], padding="VALID")
            #print conv.get_shape()
            conv = tf.nn.relu(conv + self.b1[i])  # [batch_size, sentence_length-ws+1, 1, num_filters_A]
          with tf.name_scope("pool-ws%d" %ws):
            pool = pooling(conv, axis=1)# (batch_size, 1, num_filters_A)
          pools.append(pool)
        out.append(pools)
      return out

  def bulid_block_B(self, x):
    out = []
    with tf.name_scope("bulid_block_B"):
      for pooling in self.poolings[:-1]:
        pools = []
        with tf.name_scope("conv-pool"):
          for i, ws in enumerate(self.filter_sizes[:-1]):
            with tf.name_scope("per_conv-pool-%s" % ws):
              pool = self.per_dim_conv_layer(x, self.W2[i], self.b2[i], pooling)
            pools.append(pool)
          out.append(pools)
      return out


class SentencePairEncoderCNN(SentencePairEncoder):
    def init_extra(self):
        '''
        This function could be override in a child class 
        for extra placeholders or parameters.
        '''
        self.labels = tf.placeholder(tf.int32, [None], name='labels')
        self.linear_matrix = tf.get_variable(name='linear_matrix',
                                               shape=[self._dim * 2, self._dim],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        self.score_vector = tf.get_variable(name='score_vector',
                                               shape=[self._dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        
        self.W_conv1 = tf.get_variable(name='W_conv1',
                                  shape=[2, self.word_embeddings.shape[1], 1, self._dim],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=True)
        self.b_conv1 = tf.get_variable(name='b_conv1', 
                                  initializer=tf.constant(np.array([0.1] * self._dim, dtype=np.float32)),
                                  trainable=True)
    
    def produce_feature(self, sentences, lens):
        '''
        This function could be overrided in a child class
        to generate qa pair features from word embedding layer.
        '''
        #(batchsize, seq_length, embed_size)
        embed_layer = tf.nn.embedding_lookup(self.word_embeddings, sentences)
        #(batchsize, seq_length, embed_size) 
        embed_layer_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * embed_layer
        #(batchsize, seq_length, embed_size, 1)
        embed_layer_mask = tf.expand_dims(embed_layer_mask, 3)
        #(batchsize, -1, 1, dim)
        layer_conv1 = tf.nn.conv2d(embed_layer_mask, self.W_conv1, strides=[1,1,1,1], padding="VALID",name="conv1")
        #(batchsize, -1, 1, dim)
        layer_conv1 = self._non_linear(layer_conv1 + self.b_conv1)
        #(batchsize, 1, dim)
        layer_pool1 = tf.reduce_max(layer_conv1, axis=1)
        #(batchsize, dim)
        features = tf.squeeze(layer_pool1, 1)
        
        return features

class SentencePairEncoderPairwiseRanking(SentencePairEncoder):
    def init_extra(self):
        '''
        This function could be override in a child class 
        for extra placeholders or parameters.
        '''
        self.neg_answers = tf.placeholder(tf.int32, [None, self._seq_length], name='neg_answers')
        self.neg_answer_lens = tf.placeholder(tf.int32, [None], name='neg_answer_lens')
        self.linear_matrix = tf.get_variable(name='linear_matrix',
                                               shape=[self._embed_dim * 2, self._dim],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        self.score_vector = tf.get_variable(name='score_vector',
                                               shape=[self._dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
    
    def derive_loss(self):
        '''
        This function could be overided in a child class
        to derive score and loss.
        '''
        q_features = self.produce_feature(self.input_questions, self.input_question_lens) 
        pos_a_features = self.produce_feature(self.input_answers, self.input_answer_lens) 
        neg_a_features = self.produce_feature(self.neg_answers, self.neg_answer_lens)
         
        pos_pair_features = tf.concat([q_features, pos_a_features], axis=1)
        pos_pair_features = tf.nn.relu(tf.matmul(pos_pair_features, self.linear_matrix))#(batchsize, last_dim)
        pos_pair_features = tf.nn.dropout(pos_pair_features, self.keep_prob)
        pos_scores = tf.matmul(pos_pair_features, self.score_vector)#(batchsize, 1)
        self.scores = tf.reshape(pos_scores, [-1])#(batchsize,)
        neg_pair_features = tf.concat([q_features, neg_a_features], axis=1)
        neg_pair_features = tf.nn.relu(tf.matmul(neg_pair_features, self.linear_matrix))#(batchsize, last_dim)
        neg_pair_features = tf.nn.dropout(neg_pair_features, self.keep_prob)
        neg_scores = tf.matmul(neg_pair_features, self.score_vector)#(batchsize, 1)
        neg_scores = tf.reshape(neg_scores, [-1])#(batchsize,)
        losses = tf.maximum(0.0, 1 - self.scores + neg_scores)
        self.loss = tf.reduce_mean(losses)
    
            
class SentencePairEncoderPairwiseRankingCNN(SentencePairEncoderPairwiseRanking):
    def init_extra(self):
        '''
        This function could be override in a child class 
        for extra placeholders or parameters.
        '''
        self.neg_answers = tf.placeholder(tf.int32, [None, self._seq_length], name='neg_answers')
        self.neg_answer_lens = tf.placeholder(tf.int32, [None], name='neg_answer_lens')
        self.linear_matrix = tf.get_variable(name='linear_matrix',
                                               shape=[18 * self._dim, self._dim],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        self.score_vector = tf.get_variable(name='score_vector',
                                               shape=[self._dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        self.W_conv1 = tf.get_variable(name='W_conv1',
                                  shape=[1, self.word_embeddings.shape[1], 1, self._dim],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=True)
        self.b_conv1 = tf.get_variable(name='b_conv1', 
                                  initializer=tf.constant(np.array([0.1] * self._dim, dtype=np.float32)),
                                  trainable=True)
        self.W_conv2 = tf.get_variable(name='W_conv2',
                                  shape=[2, self.word_embeddings.shape[1], 1, self._dim],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=True)
        self.b_conv2 = tf.get_variable(name='b_conv2', 
                                  initializer=tf.constant(np.array([0.1] * self._dim, dtype=np.float32)),
                                  trainable=True)
        self.W_conv3 = tf.get_variable(name='W_conv3',
                                  shape=[3, self.word_embeddings.shape[1], 1, self._dim],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  trainable=True)
        self.b_conv3 = tf.get_variable(name='b_conv3', 
                                  initializer=tf.constant(np.array([0.1] * self._dim, dtype=np.float32)),
                                  trainable=True)
    
    def produce_feature(self, sentences, lens):
        '''
        This function could be overrided in a child class
        to generate qa pair features from word embedding layer.
        '''
        #(batchsize, seq_length, embed_size)
        embed_layer = tf.nn.embedding_lookup(self.word_embeddings, sentences)
        #(batchsize, seq_length, embed_size) 
        embed_layer_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * embed_layer
        #(batchsize, seq_length, embed_size, 1)
        embed_layer_mask = tf.expand_dims(embed_layer_mask, 3)
        #(batchsize, seq_length, 1, dim)
        layer_conv1 = tf.nn.conv2d(embed_layer_mask, self.W_conv1, strides=[1,1,self.word_embeddings.shape[1],1], padding="SAME",name="conv1")
        #(batchsize, seq_length, 1, dim)
        layer_conv1 = self._non_linear(layer_conv1 + self.b_conv1)
        #(batchsize, seq_length, 1, dim)
        layer_conv2 = tf.nn.conv2d(embed_layer_mask, self.W_conv2, strides=[1,1,self.word_embeddings.shape[1],1], padding="SAME",name="conv2")
        #(batchsize, seq_length, 1, dim)
        layer_conv2 = self._non_linear(layer_conv2 + self.b_conv2)
        #(batchsize, seq_length, 1, dim)
        layer_conv3 = tf.nn.conv2d(embed_layer_mask, self.W_conv3, strides=[1,1,self.word_embeddings.shape[1],1], padding="SAME",name="conv3")
        #(batchsize, seq_length, 1, dim)
        layer_conv3 = self._non_linear(layer_conv3 + self.b_conv3)
        #(batchsize, seq_length, dim)
        layer_conv1 = tf.squeeze(layer_conv1, 2)
        layer_conv2 = tf.squeeze(layer_conv2, 2)
        layer_conv3 = tf.squeeze(layer_conv3, 2)
        #(batchsize, seq_length, dim)
        layer_conv1 = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * layer_conv1
        layer_conv2 = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * layer_conv2
        layer_conv3 = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * layer_conv3
        #(batchsize, dim)
        layer_pool1_max = tf.reduce_max(layer_conv1, axis=1)
        layer_pool1_mean = tf.reduce_mean(layer_conv1, axis=1)
        layer_pool1_min = tf.reduce_min(layer_conv1, axis=1)
        layer_pool2_max = tf.reduce_max(layer_conv2, axis=1)
        layer_pool2_mean = tf.reduce_mean(layer_conv2, axis=1)
        layer_pool2_min = tf.reduce_min(layer_conv2, axis=1)
        layer_pool3_max = tf.reduce_max(layer_conv3, axis=1)
        layer_pool3_mean = tf.reduce_mean(layer_conv3, axis=1)
        layer_pool3_min = tf.reduce_min(layer_conv3, axis=1)
        #9 * (batchsize, dim)
        features = tf.concat([layer_pool1_max, layer_pool1_mean, layer_pool1_min,
                              layer_pool2_max, layer_pool2_mean, layer_pool2_min,
                              layer_pool3_max, layer_pool3_mean, layer_pool3_min], axis=1)
        #features = [layer_pool1_max, layer_pool1_mean, layer_pool1_min,
        #            layer_pool2_max, layer_pool2_mean, layer_pool2_min,
        #            layer_pool3_max, layer_pool3_mean, layer_pool3_min]
        return features
    
    def derive_loss(self):
        '''
        This function could be overided in a child class
        to derive score and loss.
        '''
        q_features = self.produce_feature(self.input_questions, self.input_question_lens) 
        pos_a_features = self.produce_feature(self.input_answers, self.input_answer_lens) 
        neg_a_features = self.produce_feature(self.neg_answers, self.neg_answer_lens)
        '''
        pos_feature_list = []
        neg_feature_list = []
        
        for i in range(len(q_features)):
            #(batchsize, 1)
            pos_cos_dis = tf.reduce_sum(q_features[i] * pos_a_features[i], 1, keep_dims=True)
            #(batchsize, 1)
            pos_l2_dis = tf.norm(q_features[i] - pos_a_features[i], ord=2, axis=1, keep_dims=True)
            pos_l1_dis = tf.norm(q_features[i] - pos_a_features[i], ord=1, axis=1, keep_dims=True)
            pos_feature_list.extend([pos_cos_dis, pos_l2_dis, pos_l1_dis])
            neg_cos_dis = tf.reduce_sum(q_features[i] * neg_a_features[i], 1, keep_dims=True)
            #(batchsize, 1)
            neg_l2_dis = tf.norm(q_features[i] - neg_a_features[i], ord=2, axis=1, keep_dims=True)
            neg_l1_dis = tf.norm(q_features[i] - neg_a_features[i], ord=1, axis=1, keep_dims=True)
            neg_feature_list.extend([neg_cos_dis, neg_l2_dis, neg_l1_dis])
        '''
        #(batchsize, 27)
        #pos_pair_features = tf.concat(pos_feature_list, axis=1)
        pos_pair_features = tf.concat([q_features, pos_a_features], axis=1)
        pos_pair_features = tf.nn.dropout(pos_pair_features, self.keep_prob)
        pos_pair_features = tf.nn.relu(tf.matmul(pos_pair_features, self.linear_matrix))#(batchsize, last_dim)
        pos_pair_features = tf.nn.dropout(pos_pair_features, self.keep_prob)
        pos_scores = tf.matmul(pos_pair_features, self.score_vector)#(batchsize, 1)
        self.scores = tf.reshape(pos_scores, [-1])#(batchsize,)
        #(batchsize, 27)
        #neg_pair_features = tf.concat(neg_feature_list, axis=1)
        neg_pair_features = tf.concat([q_features, neg_a_features], axis=1)
        neg_pair_features = tf.nn.dropout(neg_pair_features, self.keep_prob)
        neg_pair_features = tf.nn.relu(tf.matmul(neg_pair_features, self.linear_matrix))#(batchsize, last_dim)
        neg_pair_features = tf.nn.dropout(neg_pair_features, self.keep_prob)
        neg_scores = tf.matmul(neg_pair_features, self.score_vector)#(batchsize, 1)
        neg_scores = tf.reshape(neg_scores, [-1])#(batchsize,)
        losses = tf.maximum(0.0, 1 - self.scores + neg_scores)
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.trainable_variables() if 'word_embeddings' not in w.name])
        self.loss = tf.reduce_mean(losses) + 0.01 * l2_loss

class SentencePairEncoderPairwiseRankingGRU(SentencePairEncoderPairwiseRanking):
    def init_extra(self):
        '''
        This function could be override in a child class 
        for extra placeholders or parameters.
        '''
        self.neg_answers = tf.placeholder(tf.int32, [None, self._seq_length], name='neg_answers')
        self.neg_answer_lens = tf.placeholder(tf.int32, [None], name='neg_answer_lens')
        #'''
        self.linear_matrix = tf.get_variable(name='linear_matrix',
                                               shape=[2 * self._dim, self._dim],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        self.score_vector = tf.get_variable(name='score_vector',
                                               shape=[self._dim, 1],
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               trainable=True)
        #'''
        #self.forback_combine_matrix = tf.get_variable(name='combine_matrix',
        #                                       shape=[2 * self._dim, self._dim],
        #                                       initializer=tf.contrib.layers.xavier_initializer(),
        #                                       trainable=True)
        #self.forward_rnn = tf.contrib.rnn.GRUCell(self._dim)
        forward_rnn_layers = [tf.contrib.rnn.GRUCell(self._dim) for _ in range(1)]
        self.forward_rnn = tf.contrib.rnn.MultiRNNCell(forward_rnn_layers)
        #self.backward_rnn = tf.contrib.rnn.GRUCell(self._dim)
        #backward_rnn_layers = [tf.contrib.rnn.GRUCell(self._dim) for _ in range(1)]
        #self.backward_rnn = tf.contrib.rnn.MultiRNNCell(backward_rnn_layers)
    
    def produce_feature(self, sentences, lens):
        '''
        This function could be overrided in a child class
        to generate qa pair features from word embedding layer.
        '''
        #(batchsize, seq_length, embed_size)
        embed_layer = tf.nn.embedding_lookup(self.word_embeddings, sentences)
        #(batchsize, seq_length, embed_size) 
        embed_layer_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * embed_layer
        #(batchsize, seq_length, dim)
        #outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.forward_rnn, self.backward_rnn, embed_layer_mask, sequence_length = lens, dtype=tf.float32)
        outputs, _ = tf.nn.dynamic_rnn(self.forward_rnn, embed_layer_mask, sequence_length = lens, dtype=tf.float32)
        #(batchsize, seq_length, dim)
        #forward_outputs, backward_outputs = outputs
        forward_outputs = outputs
        #(batchsize, seq_length, dim)
        forward_outputs_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * forward_outputs
        #backward_outputs_mask = tf.expand_dims(tf.cast(tf.sequence_mask(lens, self._seq_length), tf.float32), 2) * backward_outputs
        #(batchsize, seq_length, 2*dim)
        #hidden_states = tf.concat([forward_outputs_mask, backward_outputs_mask], 2)
        #hidden_states = tf.reshape(hidden_states, [-1, 2 * self._dim])
        #(batchsize, seq_length, dim)
        #hidden_states = self._non_linear(tf.matmul(hidden_states, self.forback_combine_matrix))
        #hidden_states = tf.reshape(hidden_states, [-1, self._seq_length, self._dim])
        #(batchsize, dim)
        #features = tf.reduce_sum(hidden_states, axis=1) / tf.expand_dims(tf.cast(lens, tf.float32), 1)
        features = tf.reduce_sum(forward_outputs_mask, axis=1) / tf.expand_dims(tf.cast(lens, tf.float32), 1)
        #features = tf.reduce_max(hidden_states, axis=1)
        return features
    
    def derive_loss(self):
        '''
        This function could be overided in a child class
        to derive score and loss.
        '''
        with tf.variable_scope('RNN', reuse=None):
            q_features = self.produce_feature(self.input_questions, self.input_question_lens) 
        with tf.variable_scope('RNN', reuse=True):
            pos_a_features = self.produce_feature(self.input_answers, self.input_answer_lens) 
            neg_a_features = self.produce_feature(self.neg_answers, self.neg_answer_lens)
        pos_pair_features = tf.concat([q_features, pos_a_features], axis=1)
        pos_pair_features = tf.nn.relu(tf.matmul(pos_pair_features, self.linear_matrix))#(batchsize, last_dim)
        pos_pair_features = tf.nn.dropout(pos_pair_features, self.keep_prob)
        pos_scores = tf.matmul(pos_pair_features, self.score_vector)#(batchsize, 1)
        #pos_scores = tf.reduce_sum(q_features * pos_a_features, 1, keep_dims=True)
        self.scores = tf.reshape(pos_scores, [-1])#(batchsize,)
        neg_pair_features = tf.concat([q_features, neg_a_features], axis=1)
        neg_pair_features = tf.nn.relu(tf.matmul(neg_pair_features, self.linear_matrix))#(batchsize, last_dim)
        neg_pair_features = tf.nn.dropout(neg_pair_features, self.keep_prob)
        neg_scores = tf.matmul(neg_pair_features, self.score_vector)#(batchsize, 1)
        #neg_scores = tf.reduce_sum(q_features * neg_a_features, 1, keep_dims=True)
        neg_scores = tf.reshape(neg_scores, [-1])#(batchsize,)
        losses = tf.maximum(0.0, 1 - self.scores + neg_scores)
        l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in tf.trainable_variables() if 'word_embeddings' not in w.name])
        self.loss = tf.reduce_mean(losses) + 0.01 * l2_loss
