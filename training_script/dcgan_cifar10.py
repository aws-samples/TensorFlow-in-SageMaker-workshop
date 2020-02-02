from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging

logging.getLogger().setLevel(logging.INFO)

import subprocess
import sys
import os

if os.path.isfile('requirements.txt'):
    logging.info("Installing from requirements file")
    f = open('requirements.txt')
    for line in f:
        subprocess.call([sys.executable, "-m", "pip", "install", line])

import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras import layers
import time

HEIGHT = 32
WIDTH = 32
DEPTH = 3
NUM_CLASSES = 10


def _dataset_parser(value):
    """Parse a CIFAR-10 record from value."""
    featdef = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }

    example = tf.io.parse_single_example(value, featdef)
    image = tf.io.decode_raw(example['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
        tf.float32)
    label = tf.cast(example['label'], tf.int32)
    # image = _train_preprocess_fn(image)
    return image #, tf.one_hot(label, NUM_CLASSES) (No need for the cifar10 lables)


def load_tfrecord_data(path, batch_size, buffer_size, epochs):
    print('Loading data: {}'.format(path + '/train.tfrecords'))

    dataset = tf.data.TFRecordDataset(path + '/train.tfrecords')

    dataset = dataset.repeat(epochs)
    dataset = dataset.prefetch(10)

    # Parse records.
    dataset = dataset.map(
        _dataset_parser, num_parallel_calls=10)
    # Shuffle the data
    dataset = dataset.shuffle(buffer_size=buffer_size)

    # Batch it up.
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


def make_generator_model():
    model = tf.keras.Sequential()
    # foundation for 4x4 image
    n_nodes = 256 * 4 * 4
    model.add(layers.Dense(n_nodes, input_dim=100))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Reshape((4, 4, 256)))
    # upsample to 8x8
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 16x16
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # upsample to 32x32
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # output layer
    model.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))
    assert model.output_shape == (None, 32, 32, 3)
    return model


def make_discriminator_model():
    model = tf.keras.Sequential()
    # normal
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=[HEIGHT, WIDTH, DEPTH]))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # downsample
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    # classifier
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def num_examples_per_epoch(subset='train'):
    if subset == 'train':
        return 40000
    elif subset == 'validation':
        return 10000
    elif subset == 'eval':
        return 10000
    else:
        raise ValueError('Invalid data subset "%s"' % subset)


@tf.function
def train_step(images, batch_size, noise_dim):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def generate_and_save_images(model, epoch, test_input, output_dir):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig(output_dir + '/image_at_epoch_{:04d}.png'.format(epoch))
    plt.close(fig)


def train(dataset, epochs, output_dir, batch_size, noise_dim):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, batch_size, noise_dim)

        # Produce images for the GIF as we go
        generate_and_save_images(generator,
                                 epoch + 1,
                                 seed,
                                 output_dir)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    generate_and_save_images(generator,
                             epochs,
                             seed,
                             output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--generator_learning_rate', type=float, default=1e-4)
    parser.add_argument('--discriminator_learning_rate', type=float, default=1e-4)
    parser.add_argument('--buffer_size', type=int, default=60000)
    parser.add_argument('--noise_dim', type=int, default=100)
    parser.add_argument('--num_examples_to_generate', type=int, default=16)

    # input data and model directories
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--dataset', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--num_gpus', type=int, default=os.environ.get('SM_NUM_GPUS'))

    args, _ = parser.parse_known_args()
    logging.info('Loading the data')
    # train_dataset = load_data(args.batch_size, args.buffer_size)
    train_dataset = load_tfrecord_data(args.dataset, args.batch_size, args.buffer_size, args.epochs)

    logging.info('Building the model')
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    # This method returns a helper function to compute cross entropy loss
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(args.generator_learning_rate)
    discriminator_optimizer = tf.keras.optimizers.Adam(args.discriminator_learning_rate)

    checkpoint_prefix = os.path.join(args.model_dir, "checkpoint/ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    seed = tf.random.normal([args.num_examples_to_generate, args.noise_dim])

    logging.info('Starting training')
    train(train_dataset, args.epochs, args.model_dir, args.batch_size, args.noise_dim)

    logging.info('Saving the trained model')
    generator.save(args.model_dir)
