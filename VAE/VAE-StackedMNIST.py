import os, random, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
random_seed = 100
random.seed(random_seed)
np.random.seed(random_seed)

import tensorflow as tf
import input_data
import tflib as lib
from scipy.stats import entropy
from tflib.linear import Linear
from tflib.batchnorm import Batchnorm
from tflib.deconv2d import Deconv2D
from tflib.conv2d import Conv2D
from scipy.misc import imsave as ims
from ops import *
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from Stacked_MNIST_Evaluation import MNIST, get_dist

model_number = 2
dpp_weight = 100.0
num_epochs = 10
num_eval_samples = 26000
main_dir = '/home/elfeki/Workspace/VAE_DPP/MNIST/'


def compute_diversity_loss(h_fake, h_real):
    def compute_diversity(h):
        h = tf.nn.l2_normalize(h, 1)
        Ly = tf.tensordot(h, tf.transpose(h), 1)
        eig_val, eig_vec = tf.self_adjoint_eig(Ly)
        return eig_val, eig_vec

    def normalize_min_max(eig_val):
        return tf.div(tf.subtract(eig_val, tf.reduce_min(eig_val)),
            tf.subtract(tf.reduce_max(eig_val), tf.reduce_min(eig_val))) # Min-max-Normalize Eig-Values

    fake_eig_val, fake_eig_vec = compute_diversity(h_fake)
    real_eig_val, real_eig_vec = compute_diversity(h_real)
    # Used a weighing factor to make the two losses operating in comparable ranges.
    eigen_values_loss = 0.0001 * tf.losses.mean_squared_error(labels=real_eig_val, predictions=fake_eig_val)
    eigen_vectors_loss = -tf.reduce_sum(tf.multiply(fake_eig_vec, real_eig_vec), 0)
    normalized_real_eig_val = normalize_min_max(real_eig_val)
    weighted_eigen_vectors_loss = tf.reduce_sum(tf.multiply(normalized_real_eig_val, eigen_vectors_loss))
    return eigen_values_loss + weighted_eigen_vectors_loss


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


class LatentAttention():
    def __init__(self, dpp_weight, num_eval_samples=26000):
        self.n_samples = 50000
        self.dpp_weight = dpp_weight
        self.num_eval_samples = num_eval_samples
        self.n_hidden = 500
        self.n_z = 128
        self.im_size = 28
        self.dim_h = 64
        self.batchsize = 100
        self.dim_x = 3 * self.im_size * self.im_size

        self.images = tf.placeholder(tf.float32, [None, 3*784])
        image_matrix = tf.reshape(self.images,[-1, 28, 28, 3])
        z_mean, z_stddev, h_real = self.recognition(image_matrix)
        samples = tf.random_normal([self.batchsize,self.n_z],0,1,dtype=tf.float32)
        guessed_z = z_mean + (z_stddev * samples)

        self.generated_images = self.generation(guessed_z)
        generated_flat = tf.reshape(self.generated_images, [self.batchsize, 3*28*28])

        self.dpp_loss = tf.constant(0.0)
        if self.dpp_weight > 0:
            samples_fake = tf.random_normal([self.batchsize, self.n_z], 0, 1, dtype=tf.float32)
            generated_fake = self.generation(samples_fake, reuse=True)
            _, _, h_fake = self.recognition(generated_fake, reuse=True)
            self.dpp_loss = compute_diversity_loss(h_fake, h_real)

        self.generation_loss = -tf.reduce_sum(self.images * tf.log(1e-8 + generated_flat) + (1-self.images) * tf.log(1e-8 + 1 - generated_flat),1)
        self.full_gen_loss = self.generation_loss + self.dpp_weight * self.dpp_loss

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) - tf.log(tf.square(z_stddev)) - 1,1)
        self.cost = tf.reduce_mean(self.generation_loss + self.latent_loss)
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)

        self.eval_samples = tf.random_normal([num_eval_samples, self.n_z], 0, 1, dtype=tf.float32)
        self.eval_generated_fakes = self.generation(self.eval_samples, reuse=True)

    # encoder/ discriminator
    def recognition(self, input_images, reuse=False):
        with tf.variable_scope("recognition") as scope:
            if reuse:
                scope.reuse_variables()
            # h1 = lrelu(conv2d(input_images, 1, 16, "d_h1")) # 28x28x1 -> 14x14x16
            # h2 = lrelu(conv2d(h1, 16, 32, "d_h2")) # 14x14x16 -> 7x7x32
            # h2_flat = tf.reshape(h2,[self.batchsize, 7*7*32])

            im = tf.reshape(input_images, [-1, 3, self.im_size, self.im_size])

            conv1 = Conv2D('Discriminator.1', 3, self.dim_h, 5, im, stride=2)
            out_conv1 = LeakyReLU(conv1)

            conv2 = Conv2D('Discriminator.2', self.dim_h, 2 * self.dim_h, 5, out_conv1, stride=2)
            out_conv2 = LeakyReLU(conv2)

            conv3 = Conv2D('Discriminator.3', 2 * self.dim_h, 4 * self.dim_h, 5, out_conv2, stride=2)
            out_conv3 = LeakyReLU(conv3)

            # dim_h = 64 |||| w=4, h=4, num_channel=4*64
            h2_flat = tf.reshape(out_conv3, [-1, 4 * 4 * 4 * self.dim_h])
            w_mean = Linear('Discriminator.w_mean', 4 * 4 * 4 * self.dim_h, self.n_z, h2_flat)
            w_stddev = Linear('Discriminator.w_stddev', 4 * 4 * 4 * self.dim_h, self.n_z, h2_flat)

        return w_mean, w_stddev, h2_flat

    # decoder/ generator
    def generation(self, z, reuse=False):
        with tf.variable_scope("generation") as scope:
            if reuse:
                scope.reuse_variables()
            fc1 = Linear('Generator.Input', self.n_z, 4 * 4 * 4 * self.dim_h, z)
            fc1 = Batchnorm('Generator.BN1', [0], fc1)
            fc1 = tf.nn.relu(fc1)
            out_fc1 = tf.reshape(fc1, [-1, 4 * self.dim_h, 4, 4])

            deconv1 = Deconv2D('Generator.2', 4 * self.dim_h, 2 * self.dim_h, 5, out_fc1)
            deconv1 = Batchnorm('Generator.BN2', [0, 2, 3], deconv1)
            deconv1 = tf.nn.relu(deconv1)
            out_deconv1 = deconv1[:, :, :7, :7]

            deconv2 = Deconv2D('Generator.3', 2 * self.dim_h, self.dim_h, 5, out_deconv1)
            deconv2 = Batchnorm('Generator.BN3', [0, 2, 3], deconv2)
            out_deconv2 = tf.nn.relu(deconv2)

            deconv3 = Deconv2D('Generator.5', self.dim_h, 3, 5, out_deconv2)
            out_deconv3 = tf.sigmoid(deconv3)

            return tf.reshape(out_deconv3, [-1, self.dim_x])
            # z_develop = dense(z, self.n_z, 7*7*32, scope='z_matrix')
            # z_matrix = tf.nn.relu(tf.reshape(z_develop, [self.batchsize, 7, 7, 32]))
            # h1 = tf.nn.relu(conv_transpose(z_matrix, [self.batchsize, 14, 14, 16], "g_h1"))
            # h2 = conv_transpose(h1, [self.batchsize, 28, 28, 1], "g_h2")
            # h2 = tf.nn.sigmoid(h2)

        return h2

    def inf_train_gen(self, dataset_path, BATCH_SIZE):
        ds = np.load(dataset_path).item()
        images, labels = ds['images'], ds['labels']
        ds_size = labels.shape[0]
        while True:
            for i in range(int(ds_size / BATCH_SIZE)):
                start = i * BATCH_SIZE
                end = (i + 1) * BATCH_SIZE
                yield images[start:end], labels[start:end]

    def save_fig_color(self, samples, out_path, idx):
        fig = plt.figure(figsize=(5, 5))
        gs = gridspec.GridSpec(5, 5)
        gs.update(wspace=0.05, hspace=0.05)

        for i, sample in enumerate(samples):
            ax = plt.subplot(gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            s = sample.reshape(3, 28, 28)
            s = s.transpose(1, 2, 0)
            plt.imshow(s, cmap='Greys_r')

        plt.savefig(out_path + '/{}.png'.format(str(idx).zfill(3)), bbox_inches='tight')
        plt.close(fig)

    def train(self, model_number=1, num_epochs=10):
        gen = self.inf_train_gen(main_dir+'data/stacked_train.npy', self.batchsize)
        visualization, _ = next(gen)
        run_config = tf.ConfigProto()
        run_config.gpu_options.allow_growth = True
        with tf.Session(config=run_config) as sess:
            np.random.seed(random_seed)
            tf.set_random_seed(random_seed)
            sess.run(tf.global_variables_initializer())
            for epoch in range(num_epochs):
                for idx in range(int(self.n_samples / self.batchsize)):
                    batch, _ = next(gen)
                    _, dpp_loss, gen_loss, lat_loss = sess.run((self.optimizer, self.dpp_loss, self.generation_loss, self.latent_loss), feed_dict={self.images: batch})
                    if idx % (self.n_samples - 3) == 0:
                        print "epoch %d: genloss %f latloss %f dpploss %f" % (epoch, np.mean(gen_loss), np.mean(lat_loss), np.mean(dpp_loss))
                        generated_test = sess.run(self.generated_images, feed_dict={self.images: visualization})
                        generated_test = generated_test.reshape(self.batchsize,28*28* 3)
                        self.save_fig_color(generated_test[:25], main_dir + "/Stacked_Results/", epoch)

            np.random.seed(random_seed)
            tf.set_random_seed(random_seed)
            eval_samples = np.array(sess.run(self.eval_generated_fakes)).reshape(self.num_eval_samples, 3, self.im_size ** 2)
            np.save(main_dir + 'data/evaluation_samples_' + str(model_number) + '.npy', eval_samples)


if __name__ == '__main__':
    model = LatentAttention(dpp_weight, num_eval_samples)
    model.train(model_number, num_epochs)

    samples = np.load(main_dir+'data/evaluation_samples_' + str(model_number) + '.npy')
    tf.reset_default_graph()
    tf.set_random_seed(random_seed)
    nnet = MNIST()
    digit_1 = nnet.classify(samples[:, 0])
    digit_2 = nnet.classify(samples[:, 1])
    digit_3 = nnet.classify(samples[:, 2])
    y_pred = []
    for i in range(len(digit_1)):
        y_pred.append(digit_1[i] * 100 + digit_2[i] * 10 + digit_3[i])
    x = np.unique(y_pred)

    ds = np.load(main_dir + '/data/stacked_train.npy').item()
    labels = np.array(ds['labels'])
    y_true = [np.argmax(lb) for lb in labels]
    qk = get_dist(y_true, 1000)
    pk = get_dist(y_pred, 1000)
    kl_score = entropy(pk, qk)
    print("Model: %d\t#Modes: %d, KL-score: %.3f\n\n" % (model_number, len(x), kl_score))


