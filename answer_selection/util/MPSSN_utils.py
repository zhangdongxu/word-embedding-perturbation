import tensorflow as tf

def compute_element_l1_distance(x, y):
  with tf.name_scope('element_wise_l1_distance'):
    d = tf.abs(tf.subtract(x, y))
  return d

def compute_l1_distance(x, y):
    with tf.name_scope('l1_distance'):
        d = tf.reduce_sum(tf.abs(tf.subtract(x, y)), axis=1)
        return d

def compute_euclidean_distance(x, y):
    with tf.name_scope('euclidean_distance'):
        d = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x, y)), axis=1) + 1e-12)
        return d

def compute_cosine_distance(x, y):
    with tf.name_scope('cosine_distance'):
        x_norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=1) + 1e-12)
        y_norm = tf.sqrt(tf.reduce_sum(tf.square(y), axis=1) + 1e-12)
        x_y = tf.reduce_sum(tf.multiply(x, y), axis=1) + 1e-12
        d = tf.divide(x_y, tf.multiply(x_norm, y_norm))
        return d

def comU1(x, y):
    t1 = compute_cosine_distance(x, y) #TODO
    t2 = compute_euclidean_distance(x, y)
    t3 = compute_element_l1_distance(x, y)
    result = tf.concat([tf.expand_dims(t1,1) ,tf.expand_dims(t2,1)],axis=1)
    result = tf.concat([result,t3],axis=1)
    # print 'comu1' #DEBUG
    # print result.get_shape()
    return result

def comU2(x, y):
    result = [compute_cosine_distance(x, y), compute_euclidean_distance(x, y)] #TODO
    result = tf.stack(result, axis=1)
    # print 'comu2' #DEBUG
    # print result.get_shape()
    return result
