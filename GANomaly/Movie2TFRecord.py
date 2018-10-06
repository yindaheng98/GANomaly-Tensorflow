import tensorflow as tf
import cv2
import os
import re



def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def movie2tfrecord(movie_path):
    """
    convert a movie to tfrecord
    the tfrecord's name is same as movie's name
    :param movie_path: path to movie
    :return: no return
    """
    video_capture = cv2.VideoCapture()
    video_capture.open(movie_path)
    tfrecord_path = re.split(r'\.[a-zA-Z0-9]*$', movie_path)[0] + '.tfrecord'
    with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
        success, frame = video_capture.read()
        while success:
            success, img_encode = cv2.imencode('.png', frame)  #encode image as png
            if not success:
                break
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'height': _int64_feature(frame.shape[0]),
                        'width': _int64_feature(frame.shape[1]),
                        'depth': _int64_feature(frame.shape[2]),
                        'image': _bytes_feature(img_encode.tostring())
                    }
                )
            )
            writer.write(example.SerializeToString())
            success, frame = video_capture.read()
        video_capture.release()



def _parse_function(example_proto):
    features = {'height': tf.FixedLenFeature((), tf.int64, default_value=0),
                'width': tf.FixedLenFeature((), tf.int64, default_value=0),
                'depth': tf.FixedLenFeature((), tf.int64, default_value=0),
                'image': tf.FixedLenFeature((), tf.string, default_value='')}
    parsed_features = tf.parse_single_example(example_proto, features)
    parsed_features['image'] = tf.image.decode_png(parsed_features['image'])
    return parsed_features



def dataset_from_tfrecord(tfrecord_path):
    """
    construct a data set from a tfrecord
    the tfrecord should be converted from a movie by movie2tfrecord()
    :param tfrecord_path:path to the tfrecord,can be string,string list or string Tensor
    :return:a tf.data.Dataset
    """
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function)
    return dataset



def dataset_from_movie(movie_path):
    """
    construct a data set directly from a movie
    convert the movie if tfrecord not exist
    :param movie_path: path to the movie,only string is allowed
    :return: a tf.data.Dataset
    """
    tfrecord_path = re.split(r'\.[a-zA-Z0-9]*$', movie_path)[0] + '.tfrecord'
    if not os.path.exists(tfrecord_path):
        movie2tfrecord(movie_path)
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(_parse_function)
    return dataset
