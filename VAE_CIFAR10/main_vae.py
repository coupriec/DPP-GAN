import os, shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"


import tensorflow as tf
import numpy as np

np.random.seed(0)
tf.set_random_seed(0)

from CIFAR10_VAE import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--latent_dimensions', type=int, default=5, help='latent dimensions')
parser.add_argument('--num_epochs', type=int, default=1500, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
parser.add_argument('--num_epochs_to_decay_lr', type=int, default=0, help='number of epochs to decay learning rate')
parser.add_argument('--num_train', type=int, default=1, help='number of samples to train on')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--save_epochs', type=int, default=1000, help='number of epochs to save temporary checkpoint')
parser.add_argument('--model_number', type=int, default=1, help='Model Number')
parser.add_argument('--dpp_weight', type=float, default=0.0, help='DPP Weight')

args = parser.parse_args()
our_vae = VariationalAutoencoder(latent_dimensions=args.latent_dimensions,
                                 num_epochs=args.num_epochs,
                                 learning_rate=args.learning_rate,
                                 num_epochs_to_decay_lr=args.num_epochs_to_decay_lr,
                                 num_train=args.num_train,
                                 batch_size=args.batch_size,
                                 save_epochs=args.save_epochs,
                                 model_num=args.model_number,
                                 dpp_weight=args.dpp_weight)

our_vae.train()
our_vae.incep_score()
