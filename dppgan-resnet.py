import os, h5py

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tflib as lib
import tflib.linear
import tflib.cond_batchnorm
import tflib.conv2d
import tflib.batchnorm
import tflib.layernorm
import tflib.save_images
import tflib.lsun
import tflib.inception_score

import numpy as np
import tensorflow as tf
import functools
import locale
import matplotlib.pyplot as plt

locale.setlocale(locale.LC_ALL, '')

CONDITIONAL = False  # Whether to train a conditional or unconditional model
ACGAN = False  # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ITERS = 200000  # How many iterations to train for
Eig_Val_Factor = 0.0001
MODE = 'dppgan_resnet'

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
MAIN_DIR = '/home/elfeki/Workspace/DPP-GAN_Tuan/LSUN/'
DATA_DIR = MAIN_DIR + 'data/'

BATCH_SIZE = 64  # Critic batch size
GEN_BS_MULTIPLE = 1  # Generator batch size, as a multiple of BATCH_SIZE
DIM_G = 128  # Generator dimensionality
DIM_D = 128  # Critic dimensionality
NORMALIZATION_G = True  # Use batchnorm in generator?
NORMALIZATION_D = True  # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 3072  # Number of pixels in CIFAR10 (32*32*3)
LR = 2e-4  # Initial learning rate
DECAY = False  # Whether to decay LR over learning
INCEPTION_FREQUENCY = 1000  # How frequently to calculate Inception score

ACGAN_SCALE = 1.  # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1  # How to scale generator's ACGAN loss relative to WGAN loss

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print "WARNING! Conditional model without normalization in D might be effectively unconditional!"


def nonlinearity(x):
    return tf.nn.relu(x)


def Normalize(name, inputs, labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.layernorm.Layernorm(name, [1, 2, 3], inputs)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.cond_batchnorm.Batchnorm(name, [0, 2, 3], inputs, labels=labels, n_labels=10)
        else:
            return lib.batchnorm.Batchnorm(name, [0, 2, 3], inputs, fused=True)
    else:
        return inputs


def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    return output


def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n(
        [output[:, :, ::2, ::2], output[:, :, 1::2, ::2], output[:, :, ::2, 1::2], output[:, :, 1::2, 1::2]]) / 4.
    output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    output = lib.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_1 = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2 = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample == 'up':
        conv_1 = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2 = functools.partial(lib.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = lib.conv2d.Conv2D
        conv_1 = functools.partial(lib.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2 = functools.partial(lib.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name + '.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name + '.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name + '.N2', output, labels=labels)
    output = nonlinearity(output)
    output = conv_2(name + '.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output


def OptimizedResBlockDisc1(inputs):
    conv_1 = functools.partial(lib.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2 = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False,
                             biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output


def Generator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])
    output = lib.linear.Linear('Generator.Input', 128, 4 * 4 * DIM_G, noise)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    hidden_output = tf.reshape(output, [-1, 4 * 4 * 4 * DIM_D])
    output = tf.reduce_mean(output, axis=[2, 3])
    output_wgan = lib.linear.Linear('Discriminator.Output', DIM_D, 1, output)
    output_wgan = tf.reshape(output_wgan, [-1])
    if CONDITIONAL and ACGAN:
        output_acgan = lib.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output)
        return output_wgan, output_acgan, hidden_output
    else:
        return output_wgan, None, hidden_output


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
    eigen_values_loss = Eig_Val_Factor * tf.losses.mean_squared_error(labels=real_eig_val, predictions=fake_eig_val)
    eigen_vectors_loss = -tf.reduce_sum(tf.multiply(fake_eig_vec, real_eig_vec), 0)
    normalized_real_eig_val = normalize_min_max(real_eig_val)
    weighted_eigen_vectors_loss = tf.reduce_sum(tf.multiply(normalized_real_eig_val, eigen_vectors_loss))
    return eigen_values_loss + weighted_eigen_vectors_loss, eigen_values_loss, weighted_eigen_vectors_loss, fake_eig_val, real_eig_val


def generate_image(session, frame, fixed_noise_samples):
    samples = session.run(fixed_noise_samples)
    samples = ((samples + 1.) * (255. / 2)).astype('int32')
    lib.save_images.save_images(samples.reshape((100, 3, 32, 32)),
                                MAIN_DIR + '/generated_images/' + MODE + '/samples_{}.png'.format(frame))


def cifar_get_inception_score(session, samples_100):
    all_samples = []
    for i in xrange(10):
        all_samples.append(session.run(samples_100))
    all_samples = np.concatenate(all_samples, axis=0)
    all_samples = ((all_samples + 1.) * (255.99 / 2)).astype('int32')
    all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    return lib.inception_score.get_inception_score(list(all_samples))


def inf_train_gen():
    # Dataset iterators
    train_gen, dev_gen = lib.lsun.load(BATCH_SIZE, data_dir=DATA_DIR)
    while True:
        for images, _labels in train_gen():
            yield images, _labels


def train_network():
    _iteration = tf.placeholder(tf.int32, shape=None)
    all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
    all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

    fake_data = Generator(BATCH_SIZE, all_real_labels)
    all_real_data = tf.reshape(2 * ((tf.cast(all_real_data_int, tf.float32) / 256.) - .5), [BATCH_SIZE, OUTPUT_DIM])
    all_real_data += tf.random_uniform(shape=[BATCH_SIZE, OUTPUT_DIM], minval=0., maxval=1. / 128)  # dequantize

    real_and_fake_data = tf.concat([all_real_data, fake_data], axis=0)
    real_and_fake_labels = tf.concat([all_real_labels, all_real_labels], axis=0)
    disc_all, disc_all_acgan, real_and_fake_features = Discriminator(real_and_fake_data, real_and_fake_labels)
    h_real = real_and_fake_features[:BATCH_SIZE]
    h_fake = real_and_fake_features[BATCH_SIZE:]

    disc_real = disc_all[:BATCH_SIZE]
    disc_fake = disc_all[BATCH_SIZE:]
    disc_cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake) * 0.1))
    disc_cost += tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real) * 0.9))
    disc_cost /= 2.
    disc_dppgan = disc_cost

    if CONDITIONAL and ACGAN:
        disc_acgan_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_all_acgan[:BATCH_SIZE],
                                                           labels=real_and_fake_labels[:BATCH_SIZE]))
        disc_acgan_acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.to_int32(tf.argmax(disc_all_acgan[:BATCH_SIZE], dimension=1)),
                     real_and_fake_labels[:BATCH_SIZE]), tf.float32))
        disc_acgan_fake_acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.to_int32(tf.argmax(disc_all_acgan[BATCH_SIZE:], dimension=1)),
                     real_and_fake_labels[BATCH_SIZE:]), tf.float32))
        disc_cost += ACGAN_SCALE * disc_acgan_cost
    if DECAY:
        decay = tf.maximum(0., 1. - (tf.cast(_iteration, tf.float32) / ITERS))
    else:
        decay = 1.

    n_samples = GEN_BS_MULTIPLE * BATCH_SIZE
    fake_labels = tf.cast(tf.random_uniform([n_samples]) * 10, tf.int32)
    gen_fake, gen_fake_acgan, _ = Discriminator(Generator(n_samples, fake_labels), fake_labels)

    gen_cost = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_fake, labels=tf.ones_like(gen_fake) * 0.9))
    diverstiy_cost, eigen_vals_loss, eigen_vecs_loss, fake_eig_val, real_eig_val = compute_diversity_loss(h_fake,
                                                                                                          h_real)
    gen_cost += diverstiy_cost
    gen_cost /= 2.

    if CONDITIONAL and ACGAN:
        gen_acgan_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=gen_fake, labels=fake_labels))
        gen_cost += (ACGAN_SCALE_G * gen_acgan_cost)

    gen_train_op = tf.train.AdamOptimizer(learning_rate=LR * decay, beta1=0.5).minimize(gen_cost,
                                                                                        var_list=lib.params_with_name(
                                                                                            'Generator'))
    disc_train_op = tf.train.AdamOptimizer(learning_rate=LR * decay, beta1=0.5).minimize(disc_cost,
                                                                                         var_list=lib.params_with_name(
                                                                                             'Discriminator.'))

    # Generating Qualitative Examples
    frame_i = [0]
    fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32'))
    fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)

    # Calculating inception score
    fake_labels_100 = tf.cast(tf.random_uniform([2000]) * 10, tf.int32)
    samples_100 = Generator(2000, fake_labels_100)

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as session:
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        gen = inf_train_gen()

        disc_loss_arr, gen_loss_arr, div_loss_arr = [], [], []
        eig_vals_loss_arr, eig_vecs_loss_arr = [], []
        fake_eig_vals_arr, real_eig_vals_arr = [], []
        for iteration in xrange(ITERS):
            _data, _labels = gen.next()

            _gen_cost, _div_cost, _, eigen_vals_loss_curr, eigen_vecs_loss_curr, fake_eig_val_curr, real_eig_val_curr = session.run(
                [gen_cost, diverstiy_cost, gen_train_op, eigen_vals_loss, eigen_vecs_loss, fake_eig_val, real_eig_val],
                feed_dict={_iteration: iteration, all_real_data_int: _data, all_real_labels: _labels})

            _data, _labels = gen.next()
            if CONDITIONAL and ACGAN:
                _disc_cost, _disc_dppgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ = session.run(
                    [disc_cost, disc_dppgan, disc_acgan_cost, disc_acgan_acc, disc_acgan_fake_acc, disc_train_op],
                    feed_dict={all_real_data_int: _data, all_real_labels: _labels, _iteration: iteration})
            else:
                _disc_cost, _ = session.run([disc_cost, disc_train_op],
                                            feed_dict={all_real_data_int: _data, all_real_labels: _labels,
                                                       _iteration: iteration})

            if iteration > 0 and iteration % INCEPTION_FREQUENCY == 0:
                gen_loss_arr.append(2 * _gen_cost - _div_cost)
                disc_loss_arr.append(_disc_cost)
                div_loss_arr.append(_div_cost)
                eig_vals_loss_arr.append(eigen_vals_loss_curr)
                eig_vecs_loss_arr.append(eigen_vecs_loss_curr)
                fake_eig_vals_arr.append(fake_eig_val_curr)
                real_eig_vals_arr.append(real_eig_val_curr)

                generate_image(session, iteration, fixed_noise_samples)
                inception_score = cifar_get_inception_score(session, samples_100)
                print('Iter {}:\tIncep. Score: {:.5f}'.format(iteration, np.array(inception_score[0]).mean()))

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

                saver.save(session, MAIN_DIR + 'models/' + MODE + '.ckpt', global_step=iteration)
                if inception_score[0] > 7.86:
                    saver.save(session, MAIN_DIR + 'models/' + MODE + '_' + str(iteration) + '_winner.ckpt')

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
    fake_labels = tf.cast(tf.random_uniform([n_samples]) * 10, tf.int32)
    samples = Generator(n_samples, fake_labels)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, MAIN_DIR + 'models/' + MODE + '_final.ckpt')
    inception_score = cifar_get_inception_score(session, samples)
    print MODE
    print('Evaluation Inception Score: {:.2f} +/- {:.2f}'.format(inception_score[0], inception_score[1]))


def generate_evaluation_samples(n_samples, file_name):
    np.random.seed(100)
    tf.set_random_seed(100)
    fixed_noise = tf.constant(np.random.normal(size=(n_samples, 128)).astype('float32'))
    fixed_labels = tf.constant(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 10, dtype='int32'))
    fixed_noise_samples = Generator(n_samples, fixed_labels, noise=fixed_noise)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, MAIN_DIR + 'models/' + MODE + '_final.ckpt')

    samples = session.run(fixed_noise_samples)
    samples = ((samples + 1.) * (255. / 2)).astype('int32')
    samples = samples.reshape((n_samples, 3, 32, 32))
    lib.save_images.save_images(samples, file_name)


if __name__ == "__main__":
    train_network()
    print '*' * 20
    evaluate_network(2000)
    generate_evaluation_samples(25, '/home/elfeki/Desktop/' + MODE + '_samples_1.png')
    generate_evaluation_samples(25, '/home/elfeki/Desktop/' + MODE + '_samples_2.png')
