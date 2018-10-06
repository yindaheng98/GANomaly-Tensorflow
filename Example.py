import tensorflow as tf
from GANomaly.Coders import Encoder, Decoder
from GANomaly.Movie2TFRecord import dataset_from_movie

#options
input_shape = 1024
n_hidden_channel = 32
w_loss_con = 1
w_loss_enc = 1
w_loss_adv = 1

#model
x = tf.placeholder(tf.float32, [None, input_shape, input_shape, 3])
with tf.variable_scope('encoder1'):
    z = Encoder(x, n_hidden_channel)
with tf.variable_scope('decoder1'):
    x_ = Decoder(z, n_hidden_channel)
with tf.variable_scope('encoder2'):
    z_ = Encoder(x_, n_hidden_channel)
with tf.variable_scope('encoder3'):
    fx = Encoder(x, n_hidden_channel)
with tf.variable_scope('encoder3') as scope:
    scope.reuse_variables()
    fx_ = Encoder(x_, n_hidden_channel)
predict = tf.nn.sigmoid(fx_)

#lost function
loss_con = tf.reduce_mean(tf.abs(x - x_))
loss_enc = tf.reduce_mean((z - z_) ** 2)
loss_adv = tf.reduce_mean(fx - fx_)
loss = w_loss_con * loss_con + w_loss_enc * loss_enc + w_loss_adv * loss_adv
optimizer = tf.train.AdamOptimizer(1.0).minimize(loss)

#dataset
dataset = dataset_from_movie('test/33M00S.mp4')
dataset = dataset.map(lambda e: tf.image.resize_images(e['image'], [input_shape, input_shape], method=1))
dataset = dataset.repeat()
dataset = dataset.batch(16)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

#run
with tf.Session() as sess:
    sess.run(iterator.initializer)
    sess.run(tf.global_variables_initializer())
    for i in range(0, 40):
        img = sess.run(next_element)
        l, _ = sess.run([loss, optimizer], feed_dict={x: img})
        print(l)
