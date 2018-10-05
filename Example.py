import tensorflow as tf
from GANomaly.Coders import Encoder, Decoder

input_shape = 1024
n_hidden_channel = 16
x = tf.placeholder(tf.float32, [None, input_shape, input_shape, 3])
with tf.variable_scope('encoder1'):
    z = Encoder(x, n_hidden_channel)
with tf.variable_scope('decoder1'):
    x_ = Decoder(z, n_hidden_channel, 3)
with tf.variable_scope('encoder2'):
    z_ = Encoder(x_, n_hidden_channel)
with tf.variable_scope('encoder3'):
    fx = Encoder(x, n_hidden_channel)
with tf.variable_scope('encoder3') as scope:
    scope.reuse_variables()
    fx_ = Encoder(x_, n_hidden_channel)
predict = tf.nn.sigmoid(fx_)

loss_con = tf.reduce_mean(tf.abs(x - x_))
loss_enc = tf.reduce_mean((z - z_) ** 2)
loss_adv = tf.reduce_mean(fx - fx_)

print(x.get_shape())
print(z.get_shape())
print(x_.get_shape())
print(z_.get_shape())
print(fx.get_shape())
print(fx_.get_shape())
print(predict.get_shape())
print(loss_con.get_shape())
print(loss_enc.get_shape())
print(loss_adv.get_shape())
