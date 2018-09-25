import os, sys, pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import h5py
import numpy as np
np.random.seed(1)
import tensorflow as tf
import matplotlib.pyplot as plt

import tflib as lib
from tflib.linear import Linear
from tflib.batchnorm import Batchnorm
from tflib.deconv2d import Deconv2D
from tflib.conv2d import Conv2D
from tflib.inception_score import get_inception_score
from tflib.cifar10 import load
from tflib.save_images import save_images

MAIN_DIR = '/home/elfeki/Workspace/DPP-GAN_Tuan/CIFAR/'
DATA_DIR = MAIN_DIR + 'data/cifar10/'

# Parameters to change
MODE = 'dppgan'  # Valid options are dcgan, wgan, wgan-gp, or dppgan
DIM = 128  # This overfits substantially; you're probably better off with 64
# dcgan: 0.09107 +/- 0.0402
# wgan:0.08913 +/- 0.03737
# wgan-gp: 0.08910 +/- 0.03866
# dppgan: 0.08834 +/- 0.03842
# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!

LAMBDA = 10  # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 64  # Batch size
ITERS = 50000  # How many generator iterations to train for
OUTPUT_DIM = 3072  # Number of pixels in CIFAR10 (3*32*32)

print MODE


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = Linear(name + '.Linear', n_in, n_out, inputs)
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = Linear(name + '.Linear', n_in, n_out, inputs)
    return LeakyReLU(output)


def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = Linear('Generator.Input', 128, 4 * 4 * 4 * DIM, noise)
    output = Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4 * DIM, 4, 4])

    output = Deconv2D('Generator.2', 4 * DIM, 2 * DIM, 5, output)
    output = Batchnorm('Generator.BN2', [0, 2, 3], output)
    output = tf.nn.relu(output)

    output = Deconv2D('Generator.3', 2 * DIM, DIM, 5, output)
    output = Batchnorm('Generator.BN3', [0, 2, 3], output)
    output = tf.nn.relu(output)

    output = Deconv2D('Generator.5', DIM, 3, 5, output)

    output = tf.tanh(output)

    return tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 3, 32, 32])

    output = Conv2D('Discriminator.1', 3, DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = Conv2D('Discriminator.2', DIM, 2 * DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = Batchnorm('Discriminator.BN2', [0, 2, 3], output)
    output = LeakyReLU(output)

    output = Conv2D('Discriminator.3', 2 * DIM, 4 * DIM, 5, output, stride=2)
    if MODE != 'wgan-gp':
        output = Batchnorm('Discriminator.BN3', [0, 2, 3], output)
    output = LeakyReLU(output)

    hidden_features = tf.reshape(output, [-1, 4 * 4 * 4 * DIM])
    output = Linear('Discriminator.Output', 4 * 4 * 4 * DIM, 1, hidden_features)

    return tf.reshape(output, [-1]), hidden_features


def compute_diversity_loss(h_fake, h_real):  # DPP-Loss
    def compute_diversity(h):
        h = tf.nn.l2_normalize(h, 1)
        Ly = tf.tensordot(h, tf.transpose(h), 1)
        eig_val, eig_vec = tf.self_adjoint_eig(Ly)
        eig_vec = tf.nn.l2_normalize(eig_vec, 0)
        return eig_val, eig_vec

    def normalize_min_max(eig_val):
        return tf.div(tf.subtract(eig_val, tf.reduce_min(eig_val)),
                      tf.subtract(tf.reduce_max(eig_val), tf.reduce_min(eig_val)))  # Min-max-Normalize Eig-Values

    fake_eig_val, fake_eig_vec = compute_diversity(h_fake)
    real_eig_val, real_eig_vec = compute_diversity(h_real)
    # Used a weighing factor to make the two losses operating in comparable ranges.
    eigen_values_loss = 0.0001 * tf.losses.mean_squared_error(labels=real_eig_val, predictions=fake_eig_val)
    eigen_vectors_loss = -tf.reduce_sum(tf.multiply(fake_eig_vec, real_eig_vec), 0)
    normalized_real_eig_val = normalize_min_max(real_eig_val)
    weighted_eigen_vectors_loss = tf.reduce_sum(tf.multiply(normalized_real_eig_val, eigen_vectors_loss))
    return eigen_values_loss + weighted_eigen_vectors_loss, eigen_values_loss, weighted_eigen_vectors_loss, fake_eig_val, real_eig_val


def generate_image(session, frame, fixed_noise_samples_128):
    samples = session.run(fixed_noise_samples_128)
    samples = ((samples + 1.) * (255. / 2)).astype('int32')
    save_images(samples.reshape((128, 3, 32, 32))[:],
                MAIN_DIR + '/generated_images/' + MODE + '/samples_{}.jpg'.format(frame))


def cifar_get_inception_score(session, samples_100):
    all_samples = []
    for i in xrange(10):
        all_samples.append(session.run(samples_100))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples + 1.) * (255. / 2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return get_inception_score(list(all_samples))


def inf_train_gen(batch_size=BATCH_SIZE):
    # Dataset iterators
    train_gen, dev_gen = load(batch_size, data_dir=DATA_DIR)
    while True:
        for images, _ in train_gen():
            yield images


def train_network():
    real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    real_data = 2 * ((tf.cast(real_data_int, tf.float32) / 255.) - .5)
    fake_data = Generator(BATCH_SIZE)

    disc_real, h_real = Discriminator(real_data)
    disc_fake, h_fake = Discriminator(fake_data)
    diverstiy_cost, eigen_vals_loss, eigen_vecs_loss, fake_eig_val, real_eig_val = compute_diversity_loss(h_fake, h_real)

    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')

    if MODE == 'wgan':
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        gen_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost, var_list=gen_params)
        disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost, var_list=disc_params)

        clip_ops = []
        for var in disc_params:
            clip_bounds = [-.01, .01]
            clip_ops.append(
                tf.assign(
                    var,
                    tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
                )
            )
        clip_disc_weights = tf.group(*clip_ops)

    elif MODE == 'wgan-gp':
        # Standard WGAN loss
        gen_cost = -tf.reduce_mean(disc_fake)
        disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

        # Gradient penalty
        alpha = tf.random_uniform(
            shape=[BATCH_SIZE, 1],
            minval=0.,
            maxval=1.
        )
        differences = fake_data - real_data
        interpolates = real_data + (alpha * differences)
        disc_interpolates, _ = Discriminator(interpolates)
        gradients = tf.gradients(disc_interpolates, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        disc_cost += LAMBDA * gradient_penalty

        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                                                                 var_list=gen_params)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                                                  var_list=disc_params)

    elif MODE == 'dcgan':
        gen_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake) * 0.9))
        disc_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake) * 0.1))
        disc_cost += tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real) * 0.9))
        disc_cost /= 2.

        gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                                                      var_list=lib.params_with_name(
                                                                                          'Generator'))
        disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                                                       var_list=lib.params_with_name(
                                                                                           'Discriminator.'))
    else:  # DPP-GAN
        # Standard GAN Loss
        gen_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake) * 0.9))
        disc_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake) * 0.1))
        disc_cost += tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real) * 0.9))
        disc_cost /= 2.

        # DPP Penalty
        gen_cost += diverstiy_cost
        gen_cost /= 2.

        gen_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                                                      var_list=lib.params_with_name(
                                                                                          'Generator'))
        disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                                                       var_list=lib.params_with_name(
                                                                                           'Discriminator.'))

    # For generating samples
    fixed_noise_128 = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
    fixed_noise_samples_128 = Generator(128, noise=fixed_noise_128)

    # For calculating inception score
    samples_100 = Generator(100)

    # Train loop
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    run_config.gpu_options
    with tf.Session(config=run_config) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # saver.restore(session, MAIN_DIR + 'models/wgan-gp_final.ckpt')
        gen = inf_train_gen()

        disc_loss_arr, gen_loss_arr, div_loss_arr = [], [], []
        eig_vals_loss_arr, eig_vecs_loss_arr = [], []
        fake_eig_vals_arr, real_eig_vals_arr = [], []
        mean_incep, std_incep = [], []
        for iteration in xrange(ITERS):
            _data = gen.next()
            # Train generator
            if 'wgan' not in MODE or iteration > 0:
                _gen_cost, _div_cost, _, eigen_vals_loss_curr, eigen_vecs_loss_curr, fake_eig_val_curr, real_eig_val_curr = session.run(
                    [gen_cost, diverstiy_cost, gen_train_op, eigen_vals_loss, eigen_vecs_loss, fake_eig_val, real_eig_val],
                    feed_dict={real_data_int: _data})
            # Train critic
            if 'wgan' in MODE:
                disc_iters = CRITIC_ITERS
            else:
                disc_iters = 1
            for i in xrange(disc_iters):
                if 'wgan' in MODE:
                    _data = gen.next()
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_data_int: _data})
                if MODE == 'wgan':
                    _ = session.run(clip_disc_weights)

            if iteration > 0 and iteration % 1000 == 0:
                gen_loss_arr.append(2*_gen_cost-_div_cost)
                disc_loss_arr.append(_disc_cost)
                div_loss_arr.append(_div_cost)
                eig_vals_loss_arr.append(eigen_vals_loss_curr)
                eig_vecs_loss_arr.append(eigen_vecs_loss_curr)
                fake_eig_vals_arr.append(fake_eig_val_curr)
                real_eig_vals_arr.append(real_eig_val_curr)

                generate_image(session, iteration, fixed_noise_samples_128)
                inception_score = cifar_get_inception_score(session, samples_100)
                print('Iter {}:\tIncep. Score: {:.5f}'.format(iteration, np.array(inception_score[0]).mean()))

                q = np.array(inception_score[0]).mean()
                r = np.array(inception_score[0]).std()
                mean_incep.append(q)
                std_incep.append(r)
                zipped = zip(disc_loss_arr, gen_loss_arr, div_loss_arr, mean_incep, std_incep)
                np.savetxt(MAIN_DIR + 'models/' + MODE + '_log.csv', zipped, fmt='%.5f %.5f %.5f %.5f %.5f')
                h5f = h5py.File(os.path.join(MAIN_DIR, 'models', MODE + '_eig_vals.h5'), 'w')
                h5f['eig_vals_loss'] = np.array(eig_vals_loss_arr)
                h5f['eig_vecs_loss'] = np.array(eig_vecs_loss_arr)
                h5f['fake_eig_vals'] = np.array(fake_eig_vals_arr)
                h5f['real_eig_vals'] = np.array(real_eig_vals_arr)
                h5f.close()

                plt.plot(disc_loss_arr, '-bx', label='Disc. Loss')
                plt.plot(gen_loss_arr, '-go', label='Gen. Loss')
                plt.plot(div_loss_arr, '-rd', label='Diver. Loss')
                plt.ylim([min(np.array(disc_loss_arr).min(), np.array(gen_loss_arr).min()),
                          max(np.array(disc_loss_arr).max(), np.array(gen_loss_arr).max())])
                plt.xlabel('Iteration Number')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(os.path.join(MAIN_DIR, 'models', MODE + '_training_curve_1.jpg'))
                plt.clf()

                plt.plot(disc_loss_arr, '-bx', label='Disc. Loss')
                plt.plot(gen_loss_arr, '-go', label='Gen. Loss')
                plt.plot(div_loss_arr, '-rd', label='Diver. Loss')
                plt.xlabel('Iteration Number')
                plt.ylabel('Loss')
                plt.legend()
                plt.savefig(os.path.join(MAIN_DIR, 'models', MODE + '_training_curve_2.jpg'))
                plt.gcf()
                plt.clf()

                saver.save(session, MAIN_DIR + 'models/' + MODE + '.ckpt', global_step=iteration)

        print '*' * 30
        saver.save(session, MAIN_DIR + 'models/' + MODE + '_final.ckpt')
        inception_score = cifar_get_inception_score(session, samples_100)
        print 'Final Inception Score: %.4f\n' % np.array(inception_score[0]).mean()
        zipped = zip(disc_loss_arr, gen_loss_arr, div_loss_arr)
        np.savetxt(MAIN_DIR + 'models/' + MODE + '_log.csv', zipped, fmt='%.5f %.5f %.5f')
        h5f = h5py.File(os.path.join(MAIN_DIR, 'models', MODE + '_eig_vals.h5'), 'w')
        h5f['eig_vals_loss'] = np.array(eig_vals_loss_arr)
        h5f['eig_vecs_loss'] = np.array(eig_vecs_loss_arr)
        h5f['fake_eig_vals'] = np.array(fake_eig_vals_arr)
        h5f['real_eig_vals'] = np.array(real_eig_vals_arr)
        h5f.close()

        plt.plot(disc_loss_arr, '-bx', label='Disc. Loss')
        plt.plot(gen_loss_arr, '-go', label='Gen. Loss')
        plt.plot(div_loss_arr, '-rd', label='Diver. Loss')
        plt.ylim([min(np.array(disc_loss_arr).min(), np.array(gen_loss_arr).min()),
                  max(np.array(disc_loss_arr).max(), np.array(gen_loss_arr).max())])
        plt.xlabel('Iteration Number')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(MAIN_DIR, 'models', MODE + '_training_curve_1.jpg'))
        plt.clf()

        plt.plot(disc_loss_arr, '-bx', label='Disc. Loss')
        plt.plot(gen_loss_arr, '-go', label='Gen. Loss')
        plt.plot(div_loss_arr, '-rd', label='Diver. Loss')
        plt.xlabel('Iteration Number')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(MAIN_DIR, 'models', MODE + '_training_curve_2.jpg'))
        plt.gcf()
        plt.clf()


def evaluate_network(n_samples):
    np.random.seed(1)
    tf.set_random_seed(1)
    samples = Generator(n_samples)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, MAIN_DIR + 'models/' + MODE + '_final.ckpt')
    inception_score = cifar_get_inception_score(session, samples)
    print MODE
    print('Evaluation Inception Score: {:.2f} +/- {:.2f}'.format(inception_score[0], inception_score[1]))


def evaluate_inference_via_optimization(n_samples):
    np.random.seed(1)
    tf.set_random_seed(1)
    samples = Generator(n_samples)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, MAIN_DIR + 'models/' + MODE + '_final.ckpt')

    gen = inf_train_gen()
    data = gen.next()
    data = 2 * ((data / 255.) - .5)

    sample_z_dim = 128
    z_var = tf.get_variable("z_var", shape=[n_samples, sample_z_dim], dtype=tf.float32, trainable=False)
    z_var_pl = tf.placeholder(dtype=tf.float32, shape=[n_samples, sample_z_dim], name="z_var_placeholder")
    z_var_assign = tf.assign(z_var, z_var_pl, name="z_var_assign")

    targets = data[np.random.randint(data.shape[0], size=n_samples)]

    g_loss = tf.nn.l2_loss(samples - targets)
    a = (samples + 1.) / 2.
    b = (targets + 1.) / 2.
    mse = ((a - b) ** 2)
    mse_2d = tf.reshape(mse, [n_samples, 32 * 32 * 3])
    mse = tf.reduce_mean(mse_2d, axis=1, keep_dims=True)

    optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss=g_loss, var_list=[z_var], method='L-BFGS-B',
                                                       options={'maxiter': 5, 'disp': False})

    # run the optimization from 3 different initializations
    results_images = []
    results_errors = []
    num_of_random_restarts = 3

    for i in xrange(num_of_random_restarts):
        z_sample = np.random.normal(0, 1, size=(n_samples, sample_z_dim))
        session.run(z_var_assign, {z_var_pl: z_sample})
        optimizer.minimize(session)

        generated_samples = session.run(samples)
        generated_samples_mse = session.run(mse)
        results_images.append(generated_samples)
        results_errors.append(generated_samples_mse)

    # select the best out of all random restarts
    best_images = np.zeros_like(results_images[0])
    best_images_errors = np.zeros_like(results_errors[0])
    for image_index in xrange(n_samples):
        best_img = results_images[0][image_index]
        best_img_error = results_errors[0][image_index][0]
        for indep_run_index in xrange(1, num_of_random_restarts):
            if best_img_error > results_errors[indep_run_index][image_index][0]:
                best_img_error = results_errors[indep_run_index][image_index][0]
                best_img = results_images[indep_run_index][image_index]
        best_images[image_index] = best_img
        best_images_errors[image_index][0] = best_img_error

    best_images_errors_i = np.array(best_images_errors)
    best_images_errors = np.zeros((0, 1), dtype=np.float32)
    for i in xrange(1):
        best_images_errors = np.vstack((best_images_errors, best_images_errors_i))

    if not os.path.exists('/home/elfeki/Desktop/samples'):
        os.makedirs('/home/elfeki/Desktop/samples')

    sampled = (np.array(best_images[:BATCH_SIZE]).reshape(-1, 3, 32, 32)+1)/2.
    targeted = (np.array(targets[:BATCH_SIZE]).reshape(-1, 3, 32, 32)+1)/2.
    print sampled.shape, targeted.shape
    save_images(sampled, '/home/elfeki/Desktop/samples/sampled.jpg')
    save_images(targeted, '/home/elfeki/Desktop/samples/targeted.jpg')
    print('Evaluation Inference Via Optimization: {:.5f} +/- {:.5f}'.format(np.mean(best_images_errors), np.std(best_images_errors)))


def evaluate_nearest_neighbor_number_modes(n_fake_samples):
    def unpickle(dir, file_names):
        all_data = []
        for file in file_names:
            fo = open(dir+file, 'rb')
            dict = pickle.load(fo)
            fo.close()
            all_data.extend(dict['data'])
        return np.array(all_data)

    np.random.seed(1)
    tf.set_random_seed(1)
    samples = Generator(n_fake_samples/100)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, MAIN_DIR + 'models/' + MODE + '_final.ckpt')
    for i in range(100):
        data = np.array(session.run(samples))
        pickle.dump(data, open(MAIN_DIR + 'generated_samples/'+MODE+'_batch_'+str(i), 'wb'))
        print i,
    session.close()

    # data = unpickle(DATA_DIR, ['data_batch_1', 'data_batch_2', 'data_batch_3',
    #                  'data_batch_4', 'data_batch_5', 'test_batch'])
    # data = np.array(2 * ((data / 255.) - .5))
    # print data.shape
    # marked = np.zeros(len(data))
    # file_names = []
    # for i in range(100):
    #     file_names.append(MODE+'_batch_'+str(i))
    # generated_samples = unpickle(MAIN_DIR + 'generated_samples/', file_names)
    #
    # for i in range(n_fake_samples):
    #     current_sample = np.array(generated_samples[i])
    #     min_dist = sys.float_info.max
    #     min_idx = -1
    #     for j in range(len(data)):
    #         current_real = np.array(data[j])
    #         current_dist = np.sum((current_sample-current_real)**2)
    #         if current_dist < min_dist:
    #             min_dist = current_dist
    #             min_idx = j
    #     marked[min_idx] = 1.0
    #     if i % 10000 == 0:
    #         print i/10000.0,
    # ratio = np.sum(marked) / float(len(marked))
    # print '*' * 100
    # print MODE
    # print "Ratio is: %.5f" % ratio


if __name__ == "__main__":
    train_network()
    print '*' * 20
    evaluate_network(2000)
    # evaluate_inference_via_optimization(2000)

    # evaluate_nearest_neighbor_number_modes(600000)
