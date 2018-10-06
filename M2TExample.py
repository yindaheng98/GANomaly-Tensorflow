import tensorflow as tf
import cv2

from GANomaly.Coders import Encoder, Decoder
from GANomaly.Movie2TFRecord import dataset_from_movie

#options
input_shape = 1024

#model
x = tf.placeholder(tf.float32, [None, input_shape, input_shape, 3])
z = Encoder(x)
x_ = Decoder(z)
loss = tf.reduce_mean(tf.abs(x - x_))
optimizer = tf.train.AdamOptimizer(1.0).minimize(loss)

#dataset
dataset = dataset_from_movie('test/33M00S.mp4')
dataset = dataset.map(lambda e: tf.image.resize_images(e['image'], [input_shape, input_shape], method=1))
dataset = dataset.repeat()
dataset = dataset.batch(1)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

#run
with tf.Session() as sess:
    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())
    for i in range(0, 670):
        img = sess.run(next_element)
        '''
        frame = img[0, :, :, :]
        cv2.imshow('hhh', frame[:, :, 0:3])
        cv2.waitKey(25)
        '''
        l, _ = sess.run([loss, optimizer], feed_dict={x: img})
        print(l)
