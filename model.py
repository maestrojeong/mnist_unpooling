from ops import deconvolution, convolution, get_shape, softmax_cross_entropy, leaky_relu, print_vars, extend
from configs import CAEConfig, SAVE_DIR
from tqdm import tqdm

import tensorflow as tf
import numpy as np
import logging
import os
logging.basicConfig(format = "[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")

# __name__ = '__main__'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class CAE(CAEConfig):
    def __init__(self, dataset = 'MNIST'):
        logger.info("Initialization begins")
        CAEConfig.__init__(self)
        self.layer = {}

    def build_model(self):
        logger.info("Buidling model starts...")
        self.input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, 784])
        self.layer['reshape'] = tf.reshape(self.input, [-1, 28, 28, 1])
        self.layer['conv1'] = convolution(self.layer['reshape'], [3, 3, 1, 8], padding = True, activation = tf.nn.relu, scope = "conv1")
        self.layer['conv2'] = convolution(self.layer['conv1'], [3, 3, 8, 8], padding = True, activation = tf.nn.relu, scope = "conv2")
        self.layer['pool1'] = tf.nn.max_pool(self.layer['conv2'], ksize=[1,2,2,1], strides=[1,2,2,1], padding = "SAME")
        #Store the max_pooled position
        self.layer['pool_switch1'] = tf.cast(tf.equal(self.layer['conv2'], extend(self.layer['pool1'], 2, 2)), dtype = tf.float32)
        self.layer['conv3'] = convolution(self.layer['pool1'], [3, 3, 8, 16], padding = True, activation = tf.nn.relu, scope = "conv3")
        self.layer['conv4'] = convolution(self.layer['conv3'], [3, 3, 16, 16], padding = True, activation = tf.nn.relu, scope = "conv4")
        #Auto encoded feature
        self.layer['pool2'] = tf.nn.max_pool(self.layer['conv4'], ksize=[1,2,2,1], strides=[1,2,2,1], padding = "SAME")
            #Store the max_pooled position
        self.layer['pool_switch2'] = tf.cast(tf.equal(self.layer['conv4'], extend(self.layer['pool2'], 2, 2)), dtype = tf.float32)
        self.layer['unpool2'] = extend(self.layer['pool2'], 2, 2) * self.layer['pool_switch2'] # unpooling
        self.layer['deconv4'] = deconvolution(self.layer['unpool2'], [3, 3, 16, 16], padding = True, scope = "deconv4")
        self.layer['deconv3'] = deconvolution(self.layer['deconv4'], [3, 3, 8, 16], padding = True, scope = "deconv3")
        self.layer['unpool1'] = extend(self.layer['deconv3'], 2, 2) * self.layer['pool_switch1']
        self.layer['deconv2'] = deconvolution(self.layer['unpool1'], [3, 3, 8, 8], padding = True, scope = "deconv2")
        self.layer['deconv1'] = deconvolution(self.layer['deconv2'], [3, 3, 1, 8], padding = True, activation = tf.nn.sigmoid, scope = "deconv1")
        self.layer['reconst'] = tf.reshape(self.layer['deconv1'], [-1, 784])
        print_vars("trainable_variables")
        self.square_error = tf.reduce_mean(tf.square(self.layer['reconst'] - self.input))
        self.learning_rate = tf.Variable(1e-4, trainable=False, name = "learning_rate")
        self.run_train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.square_error)
        logger.info("Buidling model done")
    
    def restore(self, sess):
        logger.info("Restoring model starts...")
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(SAVE_DIR))
        logger.info("Restoring model done.")

    def train(self, sess, image_set, label_set, epoch = None, learning_rate = None):
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 10)
        if learning_rate is not None:
            sess.run(tf.assign(self.learning_rate, learning_rate))
        
        if epoch is None:
            epoch = self.epoch

        length = len(image_set)
        nbatch = int(length/self.batch_size)

        for epoch_ in range(epoch):
            if epoch_%self.decay_every == self.decay_every-1:
                sess.run(tf.assign(self.learning_rate, sess.run(self.learning_rate)*self.decay_rate))
                logger.info("Learning rate decreases to %.5f"%sess.run(self.learning_rate))
            # shuffle start
            index = np.arange(length)
            np.random.shuffle(index)
            shuffle_image_set = image_set[index]
            shuffle_label_set = label_set[index]
            # shuffle end
            epoch_error = 0 
            for nbatch_ in tqdm(range(nbatch), ascii = True, desc = "batch"):
                _, batch_error = sess.run([self.run_train, self.square_error], feed_dict = {self.input : shuffle_image_set[self.batch_size*nbatch_:self.batch_size*(nbatch_+1)]})
                epoch_error += batch_error
            epoch_error/=nbatch
            if epoch_%self.log_every == self.log_every-1:
                print("Epoch({}/{}) train error : {}".format(epoch_+1, epoch, epoch_error))
            saver.save(sess, os.path.join(SAVE_DIR, 'model'), global_step = epoch_+1)
            logger.info("Model save in %s"%SAVE_DIR)