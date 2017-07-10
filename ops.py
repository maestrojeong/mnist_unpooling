from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from utils import struct

def mnistloader(mnist_path = "../MNIST_data"):
    '''
    Args :
        mnist_path - string
            path of mnist folder 
    '''
    mnist = input_data.read_data_sets(mnist_path, one_hot = True)
    train = struct()
    test = struct()
    val = struct()
    train.image = mnist.train.images
    train.label = mnist.train.labels
    test.image = mnist.test.images
    test.label = mnist.test.labels
    val.image = mnist.validation.images
    val.label = mnist.validation.labels
    return train, test, val

def get_shape(tensor):
    '''return the shape of tensor as list'''
    return tensor.get_shape().as_list()

def extend(tensor, hr = 2, wr = 2):
    '''
    Extend 4D tensor height, and width direction
    Args :
        tensor - NHWC(batch, height, width, channel) tensor
        hr - int
            height extenstion ratio
        wr - int
            width extenstion ratio
    return : 
        tensor - NHWC(batch, height*hr, width*wr, channel)
    '''
    batch, height, width, channel = get_shape(tensor)
    index_matrix = np.zeros((batch, hr*height, wr*width, channel, 4), dtype = np.int32)
    for b in range(batch):
        for h in range(hr*height):
            for w in range(wr*width):
                for c in range(channel):
                    index_matrix[b][h][w][c][0] = b
                    index_matrix[b][h][w][c][1] = int(h/hr)
                    index_matrix[b][h][w][c][2] = int(w/wr)
                    index_matrix[b][h][w][c][3] = c
    return tf.gather_nd(tensor, index_matrix)


def print_vars(string):
    '''print variables in collection named string'''
    print("Collection name %s"%string)
    print([v.name for v in tf.get_collection(string)])


def leaky_relu(x, leak = 0.2):
    '''Simple implementation of leaky relu'''
    f1 = 0.5*(1+leak)
    f2 = 0.5*(1-leak)
    return tf.add(tf.multiply(f1, x), tf.multiply(f2, tf.abs(x)))

def deconvolution(input_, filter_shape, output_shape = None, strides = [1,1,1,1], padding = True, activation = tf.nn.relu, batch_norm = False, istrain = False, scope = None):
    '''
    Args :
        input_ - 4D tensor
            [batch, height, width, inchannel]
        filter_shape - 1D array or list with 4 elements
            [height, width, outchannel, inchannel]
        output_shape - 1D array or list with 4 elements
            [-1, height, width, outchannel]
        strides - 1D array with elements
            default to be [1,1,1,1]
        padding - bool
            default to be True
                True 'VALID'
                False 'SAME'
        activation - activation function
            default to be tf.nn.relu
        scope - string
            default to be None
    '''
    input_shape = get_shape(input_)

    with tf.variable_scope(scope or "deconv"):
        if output_shape is None:
            output_shape = [input_shape[0], input_shape[1], input_shape[2], filter_shape[-2]]

        assert input_shape[-1]==filter_shape[-1], "inchannel value of input, and filter should be same"
        assert output_shape[-1]==filter_shape[-2], "outchannel value of output, and filter should be same" 

        if padding:
            padding = 'SAME'
        else: 
            padding = 'VALID'
        w = tf.get_variable(name="w", shape = filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)) 
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape = output_shape, strides=strides, padding=padding) 
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(deconv, center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            return activation(norm)
        else:
            b = tf.get_variable(name="b", shape = output_shape[-1], initializer=tf.constant_initializer(0.01))
            return activation(deconv + b)    

def convolution(input_, filter_shape, strides = [1,1,1,1], padding = True, activation = tf.nn.relu, batch_norm = False, istrain = False, scope = None):
    '''
    Args:
        input_ - 4D tensor
            Normally NHWC format
        filter_shape - 1D array 4 elements
            [height, width, inchannel, outchannel]
        strides - 1D array 4 elements
            default to be [1,1,1,1]
        padding - bool 
            Deteremines whether add padding or not
            True => add padding 'SAME'
            Fale => no padding  'VALID'
        activation - 
            default to be tf.nn.relu
        batch_norm - bool
            default to be False
            used to add batch-normalization
        istrain - bool
            indicate the model whether train or not
        scope - string
            default to be None
    Return:
        4D tensor
        activation(batch(conv(input_)))
    '''
    with tf.variable_scope(scope or "conv"):
        if padding:
            padding = 'SAME'
        else: 
            padding = 'VALID'
        w = tf.get_variable(name="w", shape = filter_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False)) 
        conv = tf.nn.conv2d(input_, w, strides=strides, padding=padding)
        
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(conv, center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            return activation(norm)
        else:
            b = tf.get_variable(name="b", shape = filter_shape[-1], initializer=tf.constant_initializer(0.01))
            return activation(conv + b)

def fc_layer(input_, output_size, activation = tf.nn.sigmoid, batch_norm = False, istrain = False, scope = None):
    '''fully convlolution layer'''
    with tf.variable_scope(scope or "fc"):
        w = tf.get_variable(name="w", shape = [get_shape(input_)[1], output_size], initializer=tf.contrib.layers.xavier_initializer()) 
        if batch_norm:
            norm = tf.contrib.layers.batch_norm(tf.matmul(input_, w) , center=True, scale=True, decay = 0.8, is_training=istrain, scope='batch_norm')
            return activation(norm)
        else:
            b = tf.get_variable(name="b", shape = [output_size], initializer=tf.constant_initializer(0.0))
            return activation(tf.nn.xw_plus_b(input_, w, b))

def softmax_cross_entropy(logits, labels):
    '''softmax_cross_entropy, lables : correct label logits : predicts'''
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)