import tensorflow as tf
from GANomaly.Coders import Encoder,Decoder

input_shape = 1024
x = tf.placeholder(tf.float32, [None, input_shape, input_shape, 3])
encode = Encoder(x, 16)
print(encode.get_shape())
decode = Decoder(encode, 16, 3)
print(decode.get_shape())
