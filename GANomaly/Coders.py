import tensorflow as tf



def Encoder(x, n_hidden_channel, n_extra_layer=0):
    """
    编码器
    :param x: 输入
    :param n_hidden_channel: 隐藏层通道数
    :param n_extra_layer: 额外层层数
    :return: 输出
    """
    shape = x.get_shape().as_list()[1]
    assert x.get_shape().as_list()[2] == shape, '输入网络中的图像必须是正方形'
    assert (shape & shape - 1) == 0, '输入网络中的图像的边长必须是2的整数次幂'
    assert (n_hidden_channel & n_hidden_channel - 1) == 0, '隐藏层通道数必须是2的整数次幂'

    with tf.name_scope('encode_initial_layer'):  #初始层，通道数增加到n_hidden_channel，边长减半
        conv = tf.layers.conv2d(x, filters=n_hidden_channel, kernel_size=4, strides=2, padding='same')
        relu = tf.nn.leaky_relu(conv, 0.2)
        last_layer = relu
        shape /= 2

    for i in range(1, n_extra_layer + 1):
        with tf.name_scope('encode_extra_layer_%d' % i):  #额外层，通道数不变，大小不变
            conv = tf.layers.conv2d(last_layer, filters=n_hidden_channel, kernel_size=3, strides=1, padding='same')
            batch_norm = tf.layers.batch_normalization(conv)
            relu = tf.nn.leaky_relu(batch_norm, 0.2)
            last_layer = relu

    channel = n_hidden_channel  #记录通道数，每层通道数倍增
    i_layer = 0  #记录层数，每层加一
    while shape > 4:  #处理层，一步步把数据压缩成channel*4*4
        i_layer += 1
        with tf.name_scope('encode_layer_%d' % i_layer):
            channel *= 2
            conv = tf.layers.conv2d(last_layer, channel, kernel_size=4, strides=2, padding='same')
            batch_norm = tf.layers.batch_normalization(conv)
            relu = tf.nn.leaky_relu(batch_norm, 0.2)
            last_layer = relu
            shape /= 2  #每层长宽减半

    #最后一层把数据压缩到channel*1*1
    conv = tf.layers.conv2d(last_layer,
                            filters=channel,
                            kernel_size=4, strides=1,
                            padding='valid',
                            name="encode_output_layer")
    last_layer = conv
    return last_layer



def Decoder(x, n_hidden_channel, n_output_channel, n_extra_layer=0):
    """
    解码器
    :param x: 输入
    :param n_hidden_channel: 隐藏层通道数
    :param n_extra_layer: 额外层层数
    :param output_channel: 输出通道数
    :return: 输出
    """
    xshape = x.get_shape().as_list()
    assert xshape[1:3] == [1, 1], '输入向量必须是1*1*n大小'
    assert (xshape[3] & xshape[3] - 1) == 0, '输入向量长度必须是2的整数次幂'
    assert (n_hidden_channel & n_hidden_channel - 1) == 0, '隐藏层通道数必须是2的整数次幂'

    channel = xshape[3]
    with tf.name_scope('decode_initial_layer'):
        channel //= 2
        conv_t = tf.layers.conv2d_transpose(x, channel, kernel_size=4, strides=1, padding='valid')
        batch_norm = tf.layers.batch_normalization(conv_t)
        relu = tf.nn.relu(batch_norm)
        last_layer = relu

    i_layer = 0  #记录层数，每层加一
    while channel >= n_hidden_channel:
        i_layer += 1
        with tf.name_scope('decode_layer_%d' % i_layer):  #处理层，一步步把数据恢复成n_hidden_channel*shape*shape
            channel //= 2
            conv_t = tf.layers.conv2d_transpose(last_layer, channel, kernel_size=4, strides=2, padding='same')
            batch_norm = tf.layers.batch_normalization(conv_t)
            relu = tf.nn.relu(batch_norm)
            last_layer = relu

    for i in range(1, n_extra_layer + 1):
        with tf.name_scope('decode_extra_layer_%d' % i):  #额外层，通道数不变，大小不变
            conv_t = tf.layers.conv2d_transpose(last_layer, channel, kernel_size=3, strides=1, padding='same')
            batch_norm = tf.layers.batch_normalization(conv_t)
            relu = tf.nn.relu(batch_norm)
            last_layer = relu

    #最后一层把数据恢复到channel*1*1
    conv_t = tf.layers.conv2d_transpose(last_layer,
                                        filters=n_output_channel,
                                        kernel_size=4,
                                        strides=2,
                                        padding='same',
                                        name='decode_output_layer')
    last_layer = conv_t
    return last_layer
