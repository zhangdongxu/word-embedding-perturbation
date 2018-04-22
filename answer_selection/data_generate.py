import sys
import time
import numpy as np
import tensorflow as tf
from collections import defaultdict
from random import shuffle
import pdb

from evaluator import *



class DataGeneratePointWise:
    def __init__(self, model, sess, data, batch_size, max_length, embed_dim=300, sampling = 'max', keep_prob=0.5):
        self.model = model
        self.sess = sess
        self.data = data
        self.data_size = data['size']
        self.batch_size = batch_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.keep_prob = keep_prob
    
    def padding(self, sent):
        length = len(sent)
        sent_ = sent[:]
        if length < self.max_length:
            sent_.extend([0] * (self.max_length - length))
        else:
            sent_ = sent_[:self.max_length]
        #print sent_
        return sent_
    
    def next(self, grad_noise_q, grad_noise_a, random_type=None, word_keep=1.0, weight=0.0):
        num_classes = 2
        samples = np.random.randint(self.data_size, size=self.batch_size)
        qs, qlens = [], []
        ans, alens = [], []
        labels_arr = []
        for id_ in samples:
            label = self.data['labels'][id_]
            qlen = len(self.data['lsents'][id_]) if len(self.data['lsents'][id_]) < self.max_length else self.max_length
            alen = len(self.data['rsents'][id_]) if len(self.data['rsents'][id_]) < self.max_length else self.max_length
            qs.append(self.padding(self.data['lsents'][id_]))
            qlens.append(qlen)
            ans.append(self.padding(self.data['rsents'][id_]))
            alens.append(alen)
            labels_arr.append(label)
        # To one hot
        labels_arr = np.array(labels_arr)
        labels = np.zeros((labels_arr.shape[0], num_classes))
        labels[np.arange(labels_arr.shape[0]), labels_arr] = 1
        # Add noise
        noise_q = np.ones([self.batch_size, self.max_length, self.embed_dim])
        noise_a = np.ones([self.batch_size, self.max_length, self.embed_dim])
        if random_type in ['Bernoulli', 'Bernoulli-semantic', 'Bernoulli-adversarial']:
            noise_q /= word_keep
        
        if random_type == 'Adversarial':
            feed_dict = {self.model.input_questions: np.array(qs, dtype = np.int32), 
                     self.model.input_question_lens: np.array(qlens, dtype = np.int32),
                     self.model.input_answers: np.array(ans, dtype = np.int32), 
                     self.model.input_answer_lens: np.array(alens, dtype = np.int32),
                     self.model.noise_q: noise_q,
                     self.model.noise_a: noise_a,
                     self.model.labels: np.array(labels, dtype = np.int32),
                     self.model.keep_prob: np.float32(1.0)}
            grad_noise_q_, grad_noise_a_ = self.sess.run([grad_noise_q, grad_noise_a], feed_dict=feed_dict)
        elif random_type == 'Gaussian-adversarial':
            noise_q = np.random.normal(1, weight, [self.batch_size, self.max_length, self.embed_dim])
            noise_a = np.random.normal(1, weight, [self.batch_size, self.max_length, self.embed_dim])
            feed_dict = {self.model.input_questions: np.array(qs, dtype = np.int32),
                     self.model.input_question_lens: np.array(qlens, dtype = np.int32),
                     self.model.input_answers: np.array(ans, dtype = np.int32), 
                     self.model.input_answer_lens: np.array(alens, dtype = np.int32),
                     self.model.noise_q: noise_q,
                     self.model.noise_a: noise_a,
                     self.model.labels: np.array(labels, dtype = np.int32),
                     self.model.keep_prob: np.float32(1.0)}
            grad_noise_q_, grad_noise_a_ = self.sess.run([grad_noise_q, grad_noise_a], feed_dict=feed_dict)
        elif random_type == 'Bernoulli-adversarial':
            noise_q = np.random.choice(2,size=(self.batch_size, self.max_length, self.embed_dim), p=[1-word_keep, word_keep])
            noise_a = np.random.choice(2,size=(self.batch_size, self.max_length, self.embed_dim), p=[1-word_keep, word_keep])
            feed_dict = {self.model.input_questions: np.array(qs, dtype = np.int32),
                     self.model.input_question_lens: np.array(qlens, dtype = np.int32),
                     self.model.input_answers: np.array(ans, dtype = np.int32),  
                     self.model.input_answer_lens: np.array(alens, dtype = np.int32),
                     self.model.noise_q: noise_q,
                     self.model.noise_a: noise_a,
                     self.model.labels: np.array(labels, dtype = np.int32),
                     self.model.keep_prob: np.float32(1.0)}
            grad_noise_q_, grad_noise_a_ = self.sess.run([grad_noise_q, grad_noise_a], feed_dict=feed_dict)
            number_change = (1-word_keep) * self.max_length * self.embed_dim

        for bi in range(self.batch_size):
            if random_type == 'Bernoulli':
                noise_q[bi,:,:] = np.random.choice(2,size=(self.max_length, self.embed_dim), p=[1-word_keep, word_keep])
                noise_a[bi,:,:] = np.random.choice(2,size=(self.max_length, self.embed_dim), p=[1-word_keep, word_keep])
            if random_type == 'Gaussian':
                noise_q[bi,:,:] = np.random.normal(1, weight, [self.max_length, self.embed_dim])
                noise_a[bi,:,:] = np.random.normal(1, weight, [self.max_length, self.embed_dim])
            if random_type == 'Adversarial':
                grad_noise_q_[bi] /= (np.linalg.norm(grad_noise_q_[bi]) + 1e-10)
                grad_noise_a_[bi] /= (np.linalg.norm(grad_noise_a_[bi]) + 1e-10)
                noise_q[bi,:,:] += weight * grad_noise_q_[bi]
                noise_a[bi,:,:] += weight * grad_noise_a_[bi]
            if random_type == 'Gaussian-adversarial':
                grad_noise_q_[bi] /= (np.linalg.norm(grad_noise_q_[bi]) + 1e-10)
                grad_noise_a_[bi] /= (np.linalg.norm(grad_noise_a_[bi]) + 1e-10)
                noise_q[bi,:,:] += weight * grad_noise_q_[bi]
                noise_a[bi,:,:] += weight * grad_noise_a_[bi]
            if random_type == 'Bernoulli-adversarial':
                noise_q_flat = np.reshape(noise_q[bi], [-1])
                grad_noise_q_flat = np.reshape(grad_noise_q_[bi], [-1])#(seq_length * embedding_dim)
                grad_q_flat_abs = np.fabs(grad_noise_q_flat)
                sorted_id = np.argsort(grad_q_flat_abs)
                count_change = 0
                for i in range(len(grad_noise_q_flat)):
                    id_ = sorted_id[i]
                    if count_change > number_change:
                        break
                    if noise_q_flat[id_] == 0 and grad_noise_q_flat[id_] > 0:
                        noise_q_flat[id_] = 1.0
                        count_change += 1
                    elif noise_q_flat[id_] == 1 and grad_noise_q_flat[id_] < 0:
                        noise_q_flat[id_] = 0.0
                        count_change += 1
                noise_q[bi,:,:] = np.reshape(noise_q_flat, [self.max_length, self.embed_dim])

                noise_a_flat = np.reshape(noise_a[bi], [-1])
                grad_noise_a_flat = np.reshape(grad_noise_a_[bi], [-1])#(seq_length * embedding_dim)
                grad_a_flat_abs = np.fabs(grad_noise_a_flat)
                sorted_id = np.argsort(grad_a_flat_abs)
                count_change = 0
                for i in range(len(grad_noise_a_flat)):
                    id_ = sorted_id[i]
                    if count_change > number_change:
                        break
                    if noise_a_flat[id_] == 0 and grad_noise_a_flat[id_] > 0:
                        noise_a_flat[id_] = 1.0
                        count_change += 1
                    elif noise_a_flat[id_] == 1 and grad_noise_a_flat[id_] < 0:
                        noise_a_flat[id_] = 0.0
                        count_change += 1
                noise_a[bi,:,:] = np.reshape(noise_a_flat, [self.max_length, self.embed_dim])
            if random_type == 'Bernoulli-word':
		qs[bi] = np.array(qs[bi]) * np.random.choice(2, size=self.max_length, p=[1-word_keep, word_keep])#change x by shallow copy
                ans[bi] = np.array(ans[bi]) * np.random.choice(2, size=self.max_length, p=[1-word_keep, word_keep])#change x by shallow copy
            if random_type == 'Bernoulli-semantic':
                noise_q[bi,:,:] *= np.random.choice(2, size=(self.embed_dim), p=[1-word_keep, word_keep])
                noise_a[bi,:,:] *= np.random.choice(2, size=(self.embed_dim), p=[1-word_keep, word_keep])

        feed_dict = {self.model.input_questions: np.array(qs, dtype = np.int32), 
                     self.model.input_question_lens: np.array(qlens, dtype = np.int32),
                     self.model.input_answers: np.array(ans, dtype = np.int32), 
                     self.model.input_answer_lens: np.array(alens, dtype = np.int32),
                     self.model.noise_q: noise_q,
                     self.model.noise_a: noise_a,
                     self.model.labels: np.array(labels, dtype = np.int32),
                     self.model.keep_prob: np.float32(self.keep_prob)}
        return feed_dict

class DataGeneratePairWise:
    def __init__(self, model, sess, data, batch_size, max_length, sampling='max', sample_num=5, keep_prob=0.5):
        self.model = model
        self.data = data
        self.data_size = data['size']
        self.batch_size = batch_size
        self.max_length = max_length
        self.sampling = sampling
        self.keep_prob
        self.sess = sess
    
    def padding(self, sent):
        length = len(sent)
        sent_ = sent[:]
        if length < self.max_length:
            sent_.extend([0] * (self.max_length - length))
        else:
            sent_ = sent_[:self.max_length]
        return sent_
    
    def next(self):
        pos_qs, pos_qlens = [], []
        pos_as, pos_alens = [], []
        neg_as, neg_alens = [], []
        for i in range(self.batch_size):
            while 1:
                sample_id = np.random.randint(self.data_size)
                bound, posbound, negbound = self.data['id2boundary'][sample_id]
                if posbound[0] < posbound[1] and negbound[0] < negbound[1]:
                    break
            # random
            if self.sampling != 'max':
                posid = np.random.randint(posbound[0], posbound[1])
                negid = np.random.randint(negbound[0], negbound[1])
            else:
            # min-max
                questions = np.array([self.padding(self.data['lsents'][sid]) for sid in range(negbound[0], bound[1])], dtype=np.int32)
                q_lens = np.array([len(self.data['lsents'][sid]) for sid in range(negbound[0], bound[1])], dtype=np.int32)
                answers = np.array([self.padding(self.data['rsents'][sid]) for sid in range(negbound[0], bound[1])], dtype=np.int32)
                a_lens = np.array([len(self.data['rsents'][sid]) for sid in range(negbound[0], bound[1])], dtype=np.int32)
                feed_dict = {self.model.input_questions: questions,
                         self.model.input_question_lens: q_lens,
                         self.model.input_answers: answers,
                         self.model.input_answer_lens: a_lens,
                         self.model.keep_prob: np.float32(1.0)}
                scores = self.sess.run(self.model.scores, feed_dict=feed_dict)
                posid = np.random.randint(posbound[0], posbound[1])
                negid = np.argmax(scores[:]) + negbound[0]
                #print len(scores), posbound[1]-posbound[0], negbound[1]-negbound[0]
                #posid = np.argmin(scores[:posbound[1]-posbound[0]]) + posbound[0]
                #negid = np.argmax(scores[posbound[1]-posbound[0]:]) + negbound[0]

            
            pos_qs.append(self.padding(self.data['lsents'][posid]))
            pos_qlens.append(len(self.data['lsents'][posid]))
            #print len(self.data['lsents'][posid])
            pos_as.append(self.padding(self.data['rsents'][posid]))
            pos_alens.append(len(self.data['rsents'][posid]))
            neg_as.append(self.padding(self.data['rsents'][negid]))
            neg_alens.append(len(self.data['rsents'][negid]))
        feed_dict = {self.model.input_questions: np.array(pos_qs, dtype = np.int32), 
                     self.model.input_question_lens: np.array(pos_qlens, dtype = np.int32),
                     self.model.input_answers: np.array(pos_as, dtype = np.int32), 
                     self.model.input_answer_lens: np.array(pos_alens, dtype = np.int32),
                     self.model.neg_answers: np.array(neg_as, dtype = np.int32), 
                     self.model.neg_answer_lens: np.array(neg_alens, dtype = np.int32),
                     self.model.keep_prob: np.float32(self.keep_prob)}
        return feed_dict
