import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from model import Vgg
from configs import IMAGE_DIR, NSPLITS
from utils import dataset_split, sample_img, show_gray_image, get_histogram, normalize_gray_image, get_xyhw, xyhw_visualize

# Load the dataset

# Original pictures in layers
layers = [np.squeeze(np.load(os.path.join(IMAGE_DIR, 'layer{}.npy'.format(i)))) for i in range(4)]
# Labeled by pixels in segs
segs = [np.squeeze(np.load(os.path.join(IMAGE_DIR, 'seg{}.npy'.format(i)))) for i in range(4)]

print("Each class has "+", ".join([str(lyr.shape[0]) for lyr in layers]))

normal_image = layers[0]
defect_image = np.concatenate([lyr for lyr in layers[1:4]], axis = 0)
print("Normal image : {}".format(normal_image.shape))
print("Defect image : {}".format(defect_image.shape))
nsamples, image_row, image_col = defect_image.shape

normal_train_image = layers[0][:25000]
normal_test_image = layers[0][25000:]
defect_sets = dataset_split(defect_image, NSPLITS)

sess = tf.Session()
model = Vgg(sess, 256, 256)
model.build_model()
sess.run(tf.global_variables_initializer())

index = 0
defect_train_image = np.concatenate([defect_sets[i] for i in range(NSPLITS) if i!=index ], axis = 0)
defect_test_image = defect_sets[index]

defect_train_length = len(defect_train_image)
defect_test_length = len(defect_test_image)
train_img_set = np.concatenate([defect_train_image, sample_img(normal_train_image, defect_train_length)], axis = 0)
train_label_set = np.array([0]*defect_train_length+[1]*defect_train_length)
test_img_set = np.concatenate([defect_test_image, sample_img(normal_test_image, defect_test_length)], axis = 0)
test_label_set = np.array([0]*defect_test_length+[1]*defect_test_length)

model.train(train_img_set, train_label_set, test_set=[test_img_set, test_label_set], epoch = 10)

model.get_accuracy(train_img_set, train_label_set)