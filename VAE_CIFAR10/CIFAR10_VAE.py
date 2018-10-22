from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from vae_helpers import *
import time
import os
from inception_score import get_inception_score


def compute_diversity_loss(h_fake, h_real):
    def compute_diversity(h):
        h = tf.nn.l2_normalize(h, 1)
        Ly = tf.tensordot(h, tf.transpose(h), 1)
        eig_val, eig_vec = tf.self_adjoint_eig(Ly)
        eig_vec = tf.nn.l2_normalize(eig_vec, 0)  # L2-Normalize Eig-Vectors
        return eig_val, eig_vec

    def normalize_min_max(eig_val):
        return tf.div(tf.subtract(eig_val, tf.reduce_min(eig_val)),
                      tf.subtract(tf.reduce_max(eig_val), tf.reduce_min(eig_val)))  # Min-max-Normalize Eig-Values

    fake_eig_val, fake_eig_vec = compute_diversity(h_fake)
    real_eig_val, real_eig_vec = compute_diversity(h_real)
    # Used a weighing factor to make the two losses operating in comparable ranges.
    eigen_values_loss = 0.0001 * tf.losses.mean_squared_error(labels=real_eig_val,
                                                              predictions=fake_eig_val)
    eigen_vectors_loss = -tf.reduce_sum(tf.multiply(fake_eig_vec, real_eig_vec), 0)
    normalized_real_eig_val = normalize_min_max(real_eig_val)
    weighted_eigen_vectors_loss = tf.reduce_sum(tf.multiply(normalized_real_eig_val, eigen_vectors_loss))
    return eigen_values_loss + weighted_eigen_vectors_loss


class VariationalAutoencoder(object):
    def __init__(self,
                 latent_dimensions=10,
                 num_epochs=10,
                 learning_rate=1e-3,
                 num_epochs_to_decay_lr=0,
                 num_train=50000,
                 batch_size=50,
                 save_epochs=50, model_num=1, dpp_weight=0.0):

        self.LATENTDIM = latent_dimensions
        self.NUM_EPOCS = num_epochs
        self.LEARNING_RATE = learning_rate
        self.NUM_EPOCHS_TO_DECAY_LR = num_epochs_to_decay_lr
        self.TIME_STAMP = str(model_num)
        self.dpp_weight = dpp_weight
        self.is_testing = False

        if self.NUM_EPOCHS_TO_DECAY_LR > 0:
            self.DECAY_LR = True
        else:
            self.DECAY_LR = False

        self.NUM_TRAIN = num_train
        self.BATCH_SIZE = min(num_train, batch_size)
        self.SAVE_EPOCS = save_epochs

        self.parameters = []

        with tf.name_scope('learning_parameters'):
            self.lr_placeholder = tf.placeholder("float", None, name='learning_rate')

        self._create_network()

        self._make_log_information()
        self._make_summaries()
        self.datasets = None

        self.summary_writer = tf.summary.FileWriter(self.LOG_DIR, graph=tf.get_default_graph())

    def _load_datasets(self, num_train=10):
        # load the CIFAR data -- values lie in 0-1
        mydatasets = read_cifar10_dataset('cifar-10-batches-py/')
        self.datasets = reduce_training_set(mydatasets, num_train)

    def _create_network(self):
        image_matrix = tf.placeholder("float", shape=[None, 32, 32, 3], name='x-input')
        self.x_placeholder = image_matrix
        z_mean, z_stddev, h_real = self._encoder_network(image_matrix)
        self.eps_placeholder = tf.placeholder("float", shape=[None, self.LATENTDIM])

        self.guessed_z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_stddev)), self.eps_placeholder))

        self.generated_images = self._decoder_network(self.guessed_z)

        self.dpp_loss = tf.constant(0.0)
        if self.dpp_weight > 0:
            samples_fake = tf.random_normal([self.BATCH_SIZE, self.LATENTDIM], 0, 1, dtype=tf.float32)
            self.generated_fake = self._decoder_network(samples_fake, reuse=True)
            _, _, h_fake = self._encoder_network(self.generated_fake, reuse=True)
            self.dpp_loss = compute_diversity_loss(h_fake, h_real)

        x_vectorized = tf.reshape(image_matrix, [-1, 3072], name='x-vectorized')
        x_reconstr_mean_vectorized = tf.reshape(self.x_reconstr_mean, [-1, 3072], name='x_reconstr_mean_vectorized')

        pixel_loss = tf.reduce_sum(tf.square(x_reconstr_mean_vectorized - x_vectorized), 1)
        self.pixel_loss = pixel_loss / 3072.0

        self.latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), 1)

        self.latent_loss_mean = tf.reduce_mean(self.latent_loss)
        self.pixel_loss_mean = tf.reduce_mean(self.pixel_loss)

        self.cost = tf.reduce_mean(self.latent_loss + self.pixel_loss+self.dpp_weight*self.dpp_loss, name='cost_function')  # average over batch

        self.train_step = tf.train.AdamOptimizer(learning_rate=self.lr_placeholder).minimize(self.cost,
                                                                                             var_list=self.parameters)

    def _encoder_network(self, images, reuse=False):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            # conv1_
            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32, stddev=2.0 / np.sqrt(27 + 64)),
                                 name='weights', trainable=True)
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[32], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.enc_conv1_1 = tf.nn.relu(out)
            if (not reuse and self.dpp_weight == 0) or (reuse and self.dpp_weight > 0):
                self.parameters += [kernel, biases]

            # conv1_2
            kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 32], dtype=tf.float32, stddev=2.0 / np.sqrt(144 + 64)),
                                 name='weights')
            conv = tf.nn.conv2d(self.enc_conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[32], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.enc_conv1_2 = tf.nn.relu(out)
            if (not reuse and self.dpp_weight == 0) or (reuse and self.dpp_weight > 0):
                self.parameters += [kernel, biases]

            # pool1
            self.pool1 = tf.nn.max_pool(self.enc_conv1_2,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='enc_pool1')

            # conv2_1
            kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32, stddev=2.0 / np.sqrt(144 + 16)),
                                 name='weights')
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.enc_conv2_1 = tf.nn.relu(out)
            if (not reuse and self.dpp_weight == 0) or (reuse and self.dpp_weight > 0):
                self.parameters += [kernel, biases]

            # conv2_2
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=2.0 / np.sqrt(144 + 16)),
                                 name='weights')
            conv = tf.nn.conv2d(self.enc_conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.enc_conv2_2 = tf.nn.relu(out)
            if (not reuse and self.dpp_weight == 0) or (reuse and self.dpp_weight > 0):
                self.parameters += [kernel, biases]

            # pool2
            self.pool2 = tf.nn.max_pool(self.enc_conv2_2,
                                        ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1],
                                        padding='SAME',
                                        name='enc_pool2')

            prev_dim = int(np.prod(self.pool2.get_shape()[1:]))

            pool2_flat = tf.reshape(self.pool2, [-1, prev_dim])
            # fc1
            # the total number of features extracted per input image
            fc_w = tf.Variable(tf.truncated_normal([prev_dim, self.LATENTDIM], dtype=tf.float32,
                                                   stddev=1.0 / np.sqrt(prev_dim + self.LATENTDIM)), name='weights')
            fc_b = tf.Variable(tf.constant(0.1, shape=[self.LATENTDIM], dtype=tf.float32), trainable=True,
                               name='biases')
            self.z_mean = tf.nn.bias_add(tf.matmul(pool2_flat, fc_w), fc_b)
            if (not reuse and self.dpp_weight == 0) or (reuse and self.dpp_weight > 0):
                self.parameters += [fc_w, fc_b]

            # the total number of features extracted per input image
            fc_w = tf.Variable(tf.truncated_normal([prev_dim, self.LATENTDIM], dtype=tf.float32,
                                                   stddev=2.0 / np.sqrt(prev_dim + self.LATENTDIM)), name='weights')
            fc_b = tf.Variable(tf.constant(0.1, shape=[self.LATENTDIM], dtype=tf.float32), trainable=True,
                               name='biases')
            self.z_log_sigma_sq = tf.nn.bias_add(tf.matmul(pool2_flat, fc_w), fc_b)
            if (not reuse and self.dpp_weight == 0) or (reuse and self.dpp_weight > 0):
                self.parameters += [fc_w, fc_b]

            return self.z_mean, self.z_log_sigma_sq, pool2_flat

    def _decoder_network(self, samples, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            # the size that ends up at the encoder is 8 x 8 x 64
            fc_g_w = tf.Variable(tf.truncated_normal([self.LATENTDIM, 4096], dtype=tf.float32,
                                                     stddev=1.0 / np.sqrt(self.LATENTDIM + 2048)), name='weights')
            fc_g_b = tf.Variable(tf.constant(0.1, shape=[4096], dtype=tf.float32), trainable=True, name='biases')
            a_1 = tf.nn.bias_add(tf.matmul(samples, fc_g_w), fc_g_b)
            self.g_1 = tf.nn.relu(a_1)
            if (not reuse and self.dpp_weight == 0) or (reuse and self.dpp_weight > 0):
                self.parameters += [fc_g_w, fc_g_b]

            g_1_images = tf.reshape(self.g_1, [-1, 8, 8, 64])

            # scale up to size 16 x 16 x 64
            resized_1 = None
            resized_1 = tf.image.resize_images(g_1_images, [16, 16], method=tf.image.ResizeMethod.BILINEAR)

            # conv1_1
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 32], dtype=tf.float32, stddev=2.0 / np.sqrt(288 + 16)),
                                 name='weights')
            conv = tf.nn.conv2d(resized_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.dec_conv1_1 = tf.nn.relu(out)
            if (not reuse and self.dpp_weight == 0) or (reuse and self.dpp_weight > 0):
                self.parameters += [kernel, biases]

            kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 32], dtype=tf.float32, stddev=2.0 / np.sqrt(144 + 16)),
                                 name='weights')
            conv = tf.nn.conv2d(self.dec_conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.dec_conv1_2 = tf.nn.relu(out)
            if (not reuse and self.dpp_weight == 0) or (reuse and self.dpp_weight > 0):
                self.parameters += [kernel, biases]

            # scale up to size 32 x 32 x 32
            resized_2 = tf.image.resize_images(self.dec_conv1_2, [32, 32], method=tf.image.ResizeMethod.BILINEAR)

            kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 3], dtype=tf.float32, stddev=2.0 / np.sqrt(144 + 3)),
                                 name='weights')
            conv = tf.nn.conv2d(resized_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[3], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.dec_conv2_1 = tf.nn.relu(out)
            if (not reuse and self.dpp_weight == 0) or (reuse and self.dpp_weight > 0):
                self.parameters += [kernel, biases]

            kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 3], dtype=tf.float32, stddev=2.0 / np.sqrt(27 + 3)),
                                 name='weights')
            conv = tf.nn.conv2d(self.dec_conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[3], dtype=tf.float32), trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            self.dec_conv2_2 = out
            if (not reuse and self.dpp_weight == 0) or (reuse and self.dpp_weight > 0):
                self.parameters += [kernel, biases]

            self.x_reconstr_mean = tf.sigmoid(self.dec_conv2_2)

            return self.dec_conv2_2

    def _make_log_information(self):

        # self.TIME_STAMP = get_time_stamp()
        self.LOG_DIR_ROOT = 'my_logs_dir'
        if not os.path.exists(self.LOG_DIR_ROOT):
            os.makedirs(self.LOG_DIR_ROOT)
        self.LOG_DIR = '{}/{}'.format(self.LOG_DIR_ROOT, self.TIME_STAMP)
        if not os.path.exists(self.LOG_DIR):
            os.makedirs(self.LOG_DIR)
        self.MODEL_DIR = '{}/{}'.format(self.LOG_DIR, 'model_checkpoint')
        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)
        self.TEMP_MODEL_DIR = '{}/{}'.format(self.MODEL_DIR, 'temp')
        if not os.path.exists(self.TEMP_MODEL_DIR):
            os.makedirs(self.TEMP_MODEL_DIR)
        self.RESULTS_DIR = '{}/{}'.format(self.LOG_DIR, 'results')
        if not os.path.exists(self.RESULTS_DIR):
            os.makedirs(self.RESULTS_DIR)

        self.check_point_file = '{}/model.ckpt'.format(self.MODEL_DIR)

    def _make_summaries(self):
        tf.summary.scalar('z_log_sigma_squared_min', tf.reduce_min(self.z_log_sigma_sq))
        tf.summary.scalar('z_log_sigma_squared_max', tf.reduce_max(self.z_log_sigma_sq))

        self.merged_summaries = tf.summary.merge_all()

    def train(self, num_epochs_to_display=1):
        if self.datasets is None:
            # print('loading datasets')
            self._load_datasets(num_train=self.NUM_TRAIN)

        tc = 0
        fc = 0
        lc = 0
        pc = 0

        costs = {}
        costs['latent'] = []
        costs['pixel'] = []
        costs['total'] = []

        current_lr = 1e-4

        init = tf.global_variables_initializer()
        saver = tf.train.Saver(self.parameters)

        current_epoch_cost = 0
        current_rec_cost = 0
        current_lat_cost = 0
        current_pix_cost = 0

        ITERATIONS_PER_EPOCH = int(self.NUM_TRAIN / self.BATCH_SIZE)

        with tf.Session() as sess:
            sess.run(init)

            batch_images = self.datasets.train.next_batch(self.BATCH_SIZE)[0]
            batch_images = 2 * ((batch_images / 255.) - .5)

            t0 = time.time()

            eps = np.random.normal(loc=0.0, scale=1.0, size=(self.BATCH_SIZE, self.LATENTDIM))
            tc, lc, pc, run_summary = sess.run(
                [self.cost, self.latent_loss_mean, self.pixel_loss_mean, self.merged_summaries],
                feed_dict={self.x_placeholder: batch_images, self.eps_placeholder: eps})

            # this is the initial state
            self.summary_writer.add_summary(run_summary, -1)

            # Append them to lists
            costs['total'].append(fc)
            costs['pixel'].append(pc)
            costs['latent'].append(lc)

            t1 = time.time()
            # print('Initial Cost: {:2f}  = {:.2f} L + {:.2f} P -- time taken {:.2f}'.format(tc, lc, pc, t1 - t0))

            t0 = t1

            # Train for several epochs
            for epoch in range(self.NUM_EPOCS):

                if self.DECAY_LR:
                    if epoch != 0 and epoch % self.NUM_EPOCHS_TO_DECAY_LR == 0:
                        current_lr /= 2
                        current_lr = max(current_lr, 1e-6)
                # print("learning rate : {}".format(current_lr))

                for i in range(ITERATIONS_PER_EPOCH):
                    # pick a mini batch
                    batch_images = self.datasets.train.next_batch(self.BATCH_SIZE)[0]
                    eps = np.random.normal(loc=0.0, scale=1.0, size=(self.BATCH_SIZE, self.LATENTDIM))

                    _, tc, lc, pc, run_summary = sess.run(
                        [self.train_step, self.cost, self.latent_loss_mean, self.pixel_loss_mean,
                         self.merged_summaries], feed_dict={self.x_placeholder: batch_images, self.eps_placeholder: eps,
                                                            self.lr_placeholder: current_lr})

                    current_epoch_cost += tc
                    current_lat_cost += lc
                    current_pix_cost += pc

                # for displaying costs etc -------------------------------------
                current_epoch_cost /= ITERATIONS_PER_EPOCH  # average it over the iterations
                current_lat_cost /= ITERATIONS_PER_EPOCH
                current_pix_cost /= ITERATIONS_PER_EPOCH
                # create a summary object for writing epoch costs
                summary = tf.Summary()
                summary.value.add(tag='total cost for epoch', simple_value=current_epoch_cost)
                summary.value.add(tag='latent cost for epoch', simple_value=current_lat_cost)
                summary.value.add(tag='pixel cost for epoch', simple_value=current_pix_cost)
                self.summary_writer.add_summary(summary, epoch)
                self.summary_writer.add_summary(run_summary, epoch)
                costs['total'].append(current_epoch_cost)
                costs['latent'].append(current_lat_cost)
                costs['pixel'].append(current_pix_cost)
                # --------------------------------------------------------------


                # print stats --------------------------------------------------
                if epoch % num_epochs_to_display == 0:
                    t1 = time.time()
                    # print(' epoch: {}/{} -- cost {:.2f} = {:.2f} L + {:.2f} P -- time taken {:.2f}'.format(epoch + 1,
                    #                                                                                        self.NUM_EPOCS,
                    #                                                                                        current_epoch_cost,
                    #                                                                                        current_lat_cost,
                    #                                                                                        current_pix_cost,
                    #                                                                                        t1 - t0))
                    # Reset the timer
                    t0 = t1
                # --------------------------------------------------------------

                # Reset the costs for next epoch -------------------------------
                current_epoch_cost = 0
                current_rec_cost = 0
                current_lat_cost = 0
                current_pix_cost = 0
                # --------------------------------------------------------------

                if (epoch + 1) % self.SAVE_EPOCS == 0 and epoch != 0 and epoch != (self.NUM_EPOCS - 1):
                    # Saves the weights (not the graph)
                    temp_checkpoint_file = '{}/epoch_{}.ckpt'.format(self.TEMP_MODEL_DIR, epoch)
                    save_path = saver.save(sess, temp_checkpoint_file)
                    # print("Epoch : {} Model saved in file: {}".format(epoch, save_path))
                    t0 = time.time()

            # output = np.array(sess.run(self._sample_example()))
            # print(output.shape)

            # Saves the weights (not the graph)
            save_path = saver.save(sess, self.check_point_file)
            t1 = time.time()
            # print("Model saved")

    def generate(self, z=None, n=1, checkpoint=None):
        """ Generate data from the trained model
        If z is not defined, will feed random normal as input
        """
        if z is None:
            z = np.random.random(size=(n, self.LATENTDIM))
        if checkpoint is None:
            checkpoint = self.check_point_file

        saver = tf.train.Saver(self.parameters)
        with tf.Session() as sess:
            saver.restore(sess, checkpoint)
            generated_images = sess.run(self.x_reconstr_mean, feed_dict={self.guessed_z: z})

        return generated_images

    def incep_score(self, checkpoint=None, n_samples=1000):
        checkpoint = self.check_point_file

        np.random.seed(1)
        tf.set_random_seed(1)
        checkpoint = self.check_point_file

        z = tf.random_normal([n_samples, self.LATENTDIM], 0, 1, dtype=tf.float32)
        if self.dpp_weight == 0:
            samples = self._decoder_network(z, reuse=True)
        else:
            samples = self._decoder_network(z, reuse=False)
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(self.parameters)
        saver.restore(session, checkpoint)
        generated_images = np.array(session.run(samples))
        all_samples = (((generated_images - generated_images.min()) / (generated_images.max() - generated_images.min())) * 255.).astype('int32')
        incep_score = get_inception_score(list(all_samples))
        print('Inception score: {:.5f} +/- {:.5f}'.format(incep_score[0], incep_score[1]))
        return incep_score[0], incep_score[1]

    def reconstruct_images(self, h_num=5, v_num=2):

        # show target features and estimated features
        num_visualize = h_num * v_num

        num_visualize = min(self.NUM_TRAIN, num_visualize)

        saver = tf.train.Saver(self.parameters)

        with tf.Session() as sess:
            saver.restore(sess, self.check_point_file)
            # print("Model restored.")

            sample_images = self.datasets.train.next_batch(num_visualize)[0]

            eps = np.random.normal(loc=0.0, scale=1.0, size=(num_visualize, self.LATENTDIM))
            reconstructed_images = sess.run((self.x_reconstr_mean),
                                            feed_dict={self.x_placeholder: sample_images, self.eps_placeholder: eps})

            # Reconstruct images
            plt.figure(figsize=(10, 6))
            for i in range(num_visualize):
                plt.subplot(2 * v_num, h_num, i + 1)
                plt.imshow(sample_images[i], vmin=0, vmax=1, interpolation='none', cmap=plt.get_cmap('gray'))
                plt.title("Test input")
                plt.axis('off')

                plt.subplot(2 * v_num, h_num, num_visualize + i + 1)
                plt.imshow(reconstructed_images[i], vmin=0, vmax=1, interpolation='none', cmap=plt.get_cmap('gray'))
                plt.title("Reconstruction")
                plt.axis('off')
            # plt.tight_layout()
            plt.savefig('{}/cifar-reconstruction-ld{}-bs{}-ep{}-lr{}.pdf'.format(self.RESULTS_DIR, self.LATENTDIM,
                                                                                 self.BATCH_SIZE, self.NUM_EPOCS,
                                                                                 self.LEARNING_RATE * 1e5))
            plt.close()

        ## ============= S8 Ends ==================

        # print('logs have been written to: {}'.format(self.LOG_DIR))
