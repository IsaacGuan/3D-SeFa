import os
import argparse
import scipy.misc
import numpy as np

import tensorflow as tf

from tqdm.auto import tqdm

from model import IMAE
from modelz import ZGAN


def factorize_weight(weight):
    weight = weight / np.linalg.norm(weight, axis=0, keepdims=True)
    eigen_values, eigen_vectors = np.linalg.eig(weight.dot(weight.T))

    return eigen_vectors.T, eigen_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='03001627_vox')
    args = parser.parse_args()
    dataset = args.dataset

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth=True

    tf.reset_default_graph()
    with tf.Session(config=run_config) as sess:
        zgan = ZGAN(sess, is_training=False, dataset_name=dataset, checkpoint_dir='checkpoint')
        could_load, checkpoint_counter = zgan.load(zgan.checkpoint_dir)
        boundaries, values = factorize_weight(tf.trainable_variables()[0].eval(sess))

        noise = np.random.normal(0, 0.2, [1, 128]).astype(np.float32)
        distances = np.linspace(-0.5, 0.5, 11)
        codes_list=[]
        for sem_id in range(boundaries.shape[0]):
            boundary = boundaries[sem_id:sem_id + 1]
            noises = []
            for col_id, d in enumerate(distances, start=1):
                temp_noise = noise.copy()
                temp_noise += boundary * d
                noises.append(temp_noise)
            noises = np.concatenate(noises)
            codes = sess.run(zgan.sG,
                feed_dict={zgan.z: noises}
            )
            codes_list.append(codes)

    tf.reset_default_graph()
    with tf.Session(config=run_config) as sess:
        imae = IMAE(sess, 64, 32768, is_training=False, dataset_name=dataset, checkpoint_dir='checkpoint')
        i = 0
        for sem_id in range(boundaries.shape[0]):
            sample_dir = 'samples-sefa/'+str(values[sem_id])
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            imae.test_z(codes_list[i], 64, sample_dir)
            i+=1
