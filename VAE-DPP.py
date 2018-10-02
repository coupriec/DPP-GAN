from tqdm import tqdm
import numpy as np, itertools, collections, os, random, h5py, sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
random_seed = 100
random.seed(random_seed)
np.random.seed(random_seed)
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns

# Change the write Dir to where you want to save the models & Plots
write_dir = '/home/elfeki/Workspace/DPP-GAN_ICLR/'
# Also go to the main function by the end of the function to select
# the model-number. Instructions are in the main.

if not os.path.exists(write_dir + 'models'):
    os.makedirs(write_dir + 'models')
if not os.path.exists(write_dir + 'VAE_Plots'):
    os.makedirs(write_dir + 'VAE_Plots')

params = dict(
    batch_size=512,
    learning_rate=0.001,
    max_iter=5000,
    viz_every=50,
    z_dim=256,
    x_dim=2,
    eig_vals_loss_weight=0.0001,
)

ds = tf.contrib.distributions
slim = tf.contrib.slim


def sample_ring(batch_size, n_mixture=8, std=0.01, radius=1.0):
    """Gnerate 2D Ring"""
    thetas = np.linspace(0, 2 * np.pi, n_mixture)
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
    cat = ds.Categorical(tf.zeros(n_mixture))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)
    return data.sample(batch_size, seed=random_seed)


def sample_grid(batch_size, num_components=25, std=0.05):
    """Generate 2D Grid"""
    cat = ds.Categorical(tf.zeros(num_components, dtype=tf.float32))
    mus = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                                   range(-4, 5, 2))], dtype=np.float32)
    sigmas = [np.array([std, std]).astype(np.float32) for i in range(num_components)]
    components = list((ds.MultivariateNormalDiag(mu, sigma)
                       for (mu, sigma) in zip(mus, sigmas)))
    data = ds.Mixture(cat, components)
    return data.sample(batch_size, seed=random_seed)


def evaluate_samples(generated_samples, data, model_num, is_ring_distribution=True):
    generated_samples = generated_samples[:2500]
    data = data[:2500]

    # Denormalize Data
    if is_ring_distribution:
        data = data * 4.0 - 2.0
        generated_samples = generated_samples * 4.0 - 2.0
    else:
        data = data * 6.0 - 12.0
        generated_samples = generated_samples * 6.0 - 12.0

    if is_ring_distribution:
        thetas = np.linspace(0, 2 * np.pi, 8)
        xs, ys = np.sin(thetas), np.cos(thetas)
        MEANS = np.stack([xs, ys]).transpose()
        std = 0.01
    else:  # Grid Distribution
        MEANS = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                                         range(-4, 5, 2))], dtype=np.float32)
        std = 0.05

    l2_store = []
    for x_ in generated_samples:
        l2_store.append([np.sum((x_ - i) ** 2) for i in MEANS])

    mode = np.argmin(l2_store, 1).flatten().tolist()
    dis_ = [l2_store[j][i] for j, i in enumerate(mode)]
    mode_counter = [mode[i] for i in range(len(mode)) if np.sqrt(dis_[i]) <= (3 * std)]

    sns.set(font_scale=2)
    f, (ax1, ax2) = plt.subplots(2, figsize=(10, 15))
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    sns.kdeplot(generated_samples[:, 0], generated_samples[:, 1], cmap=cmap, ax=ax1, n_levels=100, shade=True,
                clip=[[-6, 6]] * 2)
    sns.kdeplot(data[:, 0], data[:, 1], cmap=cmap, ax=ax2, n_levels=100, shade=True, clip=[[-6, 6]] * 2)

    plt.figure(figsize=(5, 5))
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], edgecolor='none')
    plt.scatter(data[:, 0], data[:, 1], c='g', edgecolor='none')
    plt.axis('off')
    plt.savefig(write_dir + 'VAE_Plots/VAE_{}_Final.png'.format(model_num))
    plt.clf()

    high_quality_ratio = np.sum(collections.Counter(mode_counter).values()) / 2500.0
    print('Model: %d || Number of Modes Captured: %d' % (model_num, len(collections.Counter(mode_counter))))
    print('Percentage of Points Falling Within 3 std. of the Nearest Mode %f' % high_quality_ratio)


def compute_diversity_loss(h_fake, h_real):
    def compute_diversity(h):
        h = tf.nn.l2_normalize(h, 1)
        Ly = tf.tensordot(h, tf.transpose(h), 1)
        eig_val, eig_vec = tf.self_adjoint_eig(Ly)
        return eig_val, eig_vec

    def normalize_min_max(eig_val):
        return tf.div(tf.subtract(eig_val, tf.reduce_min(eig_val)),
                      tf.subtract(tf.reduce_max(eig_val), tf.reduce_min(eig_val)))  # Min-max-Normalize Eig-Values

    fake_eig_val, fake_eig_vec = compute_diversity(h_fake)
    real_eig_val, real_eig_vec = compute_diversity(h_real)
    # Used a weighing factor to make the two losses operating in comparable ranges.
    eigen_values_loss = params['eig_vals_loss_weight'] * tf.losses.mean_squared_error(labels=real_eig_val,
                                                                                      predictions=fake_eig_val)
    eigen_vectors_loss = -tf.reduce_sum(tf.multiply(fake_eig_vec, real_eig_vec), 0)
    normalized_real_eig_val = normalize_min_max(real_eig_val)
    weighted_eigen_vectors_loss = tf.reduce_sum(tf.multiply(normalized_real_eig_val, eigen_vectors_loss))
    return eigen_values_loss + weighted_eigen_vectors_loss, fake_eig_val, real_eig_val, eigen_values_loss, weighted_eigen_vectors_loss


# Discriminator
def encoder(x, n_hidden=128, reuse=False):
    with tf.variable_scope("encoder", reuse=reuse):
        h_1 = slim.fully_connected(x, n_hidden, activation_fn=tf.nn.tanh)
        h_2 = slim.fully_connected(h_1, n_hidden, activation_fn=None)
        h = tf.nn.tanh(h_2)
        # log_d = slim.fully_connected(h, 1, activation_fn=None)

        w_mean = slim.fully_connected(h, params['z_dim'], activation_fn=None)
        w_stddev = slim.fully_connected(h, params['z_dim'], activation_fn=None)
    return w_mean, w_stddev, h


# Generator
def decoder(z, output_dim=params['x_dim'], n_hidden=128, n_layer=2):
    with tf.variable_scope("decoder"):
        h = slim.stack(z, slim.fully_connected, [n_hidden] * n_layer,
                       activation_fn=tf.nn.tanh)
        x = slim.fully_connected(h, output_dim, activation_fn=tf.nn.sigmoid)
    return x


def run_model(model_num):
    is_dpp_vae = False
    if model_num >= 100:
        is_dpp_vae = True

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.set_random_seed(random_seed)

        if model_num % 2 == 0:
            data = (sample_ring(params['batch_size']) + 2.0) / 4.0
        else:
            data = (sample_grid(params['batch_size']) + 6.0) / 12.0

        fake_data = ds.Normal(tf.zeros(params['x_dim']), tf.ones(params['x_dim'])).sample(params['batch_size'],
                                                                                          seed=random_seed)
        samples = ds.Normal(tf.zeros(params['z_dim']), tf.ones(params['z_dim'])).sample(params['batch_size'],
                                                                                        seed=random_seed)
        with slim.arg_scope([slim.fully_connected], weights_initializer=tf.orthogonal_initializer(gain=1.4)):
            z_mean, z_stddev, h_real = encoder(data)
            _, _, h_fake = encoder(fake_data)
            guessed_z = z_mean + (z_stddev * samples)
            generated_samples = decoder(guessed_z)

        generation_loss = -tf.reduce_sum(data * tf.log(1e-8 + generated_samples) +
                                         (1 - data) * tf.log(1e-8 + 1 - generated_samples), 1)
        latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.square(z_stddev) -
                                          tf.log(tf.square(z_stddev)) - 1, 1)
        if not is_dpp_vae:
            cost = tf.reduce_mean(generation_loss + latent_loss)
        else:
            diversity_loss, fake_lambda, real_lambda, eig_vals_loss, eig_vecs_loss = compute_diversity_loss(h_fake,
                                                                                                            h_real)
            div_loss = -tf.reduce_sum(diversity_loss)
            cost = tf.reduce_mean(generation_loss + latent_loss + div_loss)

        optimizer = tf.train.AdamOptimizer(params['learning_rate']).minimize(cost)

        sess = tf.InteractiveSession(config=run_config)
        sess.run(tf.global_variables_initializer())

        fake_lmda_list, real_lmda_list, diversity_loss_arr, gen_loss_arr, disc_loss_arr = [], [], [], [], []
        eig_vals_loss_arr, eig_vecs_loss_arr = [], []
        for idx in tqdm(xrange(params['max_iter'])):
            if not is_dpp_vae:
                _, g_loss, lat_loss = sess.run((optimizer, generation_loss, latent_loss))
            else:
                _, g_loss, lat_loss, d_loss, f_lmda, r_lmda, eig_vals_loss_curr, \
                eig_vecs_loss_curr = sess.run((optimizer, generation_loss, latent_loss, div_loss,
                                               fake_lambda, real_lambda, eig_vals_loss, eig_vecs_loss))

            if idx % params['viz_every'] == 0:
                xx, yy = sess.run([generated_samples, data])
                if model_num % 2 == 0:
                    xx = xx * 4.0 - 2.0
                    yy = yy * 4.0 - 2.0
                else:
                    xx = xx * 6.0 - 12.0
                    yy = yy * 6.0 - 12.0
                plt.figure(figsize=(5, 5))
                plt.scatter(xx[:, 0], xx[:, 1], edgecolor='none')
                plt.scatter(yy[:, 0], yy[:, 1], c='g', edgecolor='none')
                plt.axis('off')
                plt.savefig(write_dir + 'VAE_Plots/VAE_{}.png'.format(model_num))
                plt.clf()
                plt.close()

                if is_dpp_vae:
                    fake_lmda_list.append(f_lmda)
                    real_lmda_list.append(r_lmda)
                    eig_vals_loss_arr.append(eig_vals_loss_curr)
                    eig_vecs_loss_arr.append(eig_vecs_loss_curr)
                    diversity_loss_arr.append(d_loss)
                gen_loss_arr.append(g_loss)
                disc_loss_arr.append(lat_loss)

                h5f = h5py.File(write_dir + 'VAE_Plots/VAE_{}.h5'.format(model_num), 'w')
                h5f['fake_lmda'] = np.array(fake_lmda_list)
                h5f['real_lmda'] = np.array(real_lmda_list)
                h5f['diversity_loss'] = np.array(diversity_loss_arr)
                h5f['gen_loss'] = np.array(gen_loss_arr)
                h5f['disc_loss'] = np.array(disc_loss_arr)
                h5f['eig_vals_loss'] = np.array(eig_vals_loss_arr)
                h5f['eig_vecs_loss'] = np.array(eig_vecs_loss_arr)
                h5f.close()

        xx = np.vstack([sess.run(generated_samples) for _ in range(5)])
        yy = np.vstack([sess.run(data) for _ in range(5)])
        evaluate_samples(xx, yy, model_num, is_ring_distribution=(model_num % 2 == 0))
        saver = tf.train.Saver()
        saver.save(sess, write_dir + "models/VAE_{}.ckpt".format(model_num))


if __name__ == '__main__':
    # Specify a model id to experiment with.
    # Even numbers for 2D Ring, Odd numbers for 2D Grid
    # If 100 or more will be GDPP-VAE, else would be standard VAE
    run_model(150)
