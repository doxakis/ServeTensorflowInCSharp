import os
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

graph_def = tf.GraphDef()

# Import the TF graph
with tf.gfile.FastGFile('..\\model\\inception_v3.pb', 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# Load from a file
image = tf.gfile.FastGFile("..\\files\\299x299.jpg", 'rb').read()

with tf.Session() as sess:
    prob_tensor = sess.graph.get_tensor_by_name('DecodeJpeg:0')
    pixels = sess.run(prob_tensor, {'DecodeJpeg/contents:0': image })

with open('..\\results\\tensorflow.txt', 'w') as f:
    for line in pixels:
        for col in line:
            for pixel in col:
                f.write(str(pixel) + ' ')
            f.write('\n')
