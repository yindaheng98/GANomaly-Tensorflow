import tensorflow as tf



def Encoder(x, n_hidden_channel=16, n_extra_layer=0):
    """
    编码器
    :param x: 输入
    :param n_hidden_channel: 隐藏层通道数
    :param n_extra_layer: 额外层层数
    :return: 输出
    """
    shape = x.get_shape().as_list()
    channel = x.get_shape().as_list()[3]
    shape = shape[1]
    assert x.get_shape().as_list()[2] == shape, '输入网络中的图像必须是正方形'
    assert (shape & shape - 1) == 0, '输入网络中的图像的边长必须是2的整数次幂'
    assert (n_hidden_channel & n_hidden_channel - 1) == 0, '隐藏层通道数必须是2的整数次幂'

    #初始层，通道数增加到n_hidden_channel，边长减半
    w_conv = tf.get_variable('encoder_wconv_initial', shape=[4, 4, channel, n_hidden_channel],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv = tf.nn.conv2d(x, filter=w_conv, strides=[1, 2, 2, 1], padding='SAME')
    relu = tf.nn.leaky_relu(conv, 0.2)
    last_layer = relu
    shape /= 2

    #额外层，通道数不变，大小不变
    for i in range(1, n_extra_layer + 1):
        w_conv = tf.get_variable('encoder_extra_wconv_%d' % i, shape=[3, 3, n_hidden_channel, n_hidden_channel],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(last_layer, filter=w_conv, strides=[1, 1, 1, 1], padding='SAME')
        batch_norm = tf.layers.batch_normalization(conv)
        relu = tf.nn.leaky_relu(batch_norm, 0.2)
        last_layer = relu

    #处理层，一步步把数据压缩成channel*4*4
    i_layer = 0  #记录层数
    channel = n_hidden_channel  #记录通道数，每层通道数倍增
    while shape > 4:
        i_layer += 1
        channel *= 2
        w_conv = tf.get_variable('encoder_wconv_%d' % i_layer, shape=[4, 4, channel // 2, channel],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(last_layer, filter=w_conv, strides=[1, 2, 2, 1], padding='SAME')
        batch_norm = tf.layers.batch_normalization(conv)
        relu = tf.nn.leaky_relu(batch_norm, 0.2)
        last_layer = relu
        shape /= 2  #每层长宽减半

    #最后一层把数据压缩到channel*1*1
    w_conv = tf.get_variable('encoder_w_conv_final', shape=[4, 4, channel, channel],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv = tf.nn.conv2d(last_layer, filter=w_conv, strides=[1, 1, 1, 1], padding='VALID')

    last_layer = conv
    return last_layer



def Decoder(x, n_hidden_channel=16, n_output_channel=3, n_extra_layer=0):
    """
    解码器
    :param x: 输入
    :param n_hidden_channel: 隐藏层通道数
    :param n_output_channel: 输出通道数
    :param n_extra_layer: 额外层层数
    :return: 输出
    """
    xshape = x.get_shape().as_list()
    assert xshape[1:3] == [1, 1], '输入向量必须是1*1*n大小'
    assert (xshape[3] & xshape[3] - 1) == 0, '输入向量长度必须是2的整数次幂'
    assert (n_hidden_channel & n_hidden_channel - 1) == 0, '隐藏层通道数必须是2的整数次幂'

    shape = 4  #为tf.nn.conv2d_transpose中那个无用的参数output_shape记录图像边长
    channel = xshape[3]
    w_conv = tf.get_variable('decoder_wconv_initial', shape=[4, 4, channel, channel],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv = tf.nn.conv2d_transpose(x, filter=w_conv, strides=[1, 1, 1, 1], padding='SAME',
                                  output_shape=[-1, shape, shape, channel])
    batch_norm = tf.layers.batch_normalization(conv)
    relu = tf.nn.relu(batch_norm)
    last_layer = relu

    #处理层，一步步把数据恢复成n_hidden_channel*shape*shape
    i_layer = 0  #记录层数
    while channel > n_hidden_channel:
        i_layer += 1
        shape *= 2
        channel //= 2
        w_conv = tf.get_variable('decoder_wconv_%d' % i_layer, shape=[4, 4, channel, channel * 2],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d_transpose(last_layer, filter=w_conv, strides=[1, 2, 2, 1], padding='SAME',
                                      output_shape=[-1, shape, shape, channel])
        batch_norm = tf.layers.batch_normalization(conv)
        relu = tf.nn.relu(batch_norm)
        last_layer = relu

    #额外层，通道数不变，大小不变
    for i in range(1, n_extra_layer + 1):
        w_conv = tf.get_variable('decoder_extra_wconv_%d' % i, shape=[3, 3, channel, channel],
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d_transpose(last_layer, filter=w_conv, strides=[1, 1, 1, 1], padding='SAME',
                                      output_shape=[-1, shape, shape, channel])
        batch_norm = tf.layers.batch_normalization(conv)
        relu = tf.nn.relu(batch_norm)
        last_layer = relu

    #最后一层把数据恢复到channel*1*1
    shape *= 2
    w_conv = tf.get_variable('decoder_w_conv_final', shape=[4, 4, n_output_channel, channel],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))
    conv = tf.nn.conv2d_transpose(last_layer, filter=w_conv, strides=[1, 2, 2, 1], padding='SAME',
                                  output_shape=[-1, shape, shape, n_output_channel])
    tanh = tf.nn.tanh(conv)
    last_layer = tanh
    return last_layer
