import os
import pathlib
import time
import datetime
import numpy as np
from keras import Model, Sequential
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization, ReLU, LeakyReLU, Input
from tensorflow.io import read_file, decode_jpeg
import tensorflow as tf
import matplotlib.pyplot as plt


BW_PATH = pathlib.Path(__file__).parent / "kaggleData" / "bw"
COLOR_PATH = pathlib.Path(__file__).parent / "kaggleData" / "color"
BATCH_SIZE = 1
LAMBDA = 100

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def load(image_file):
    """Read and decode an image file to a uint8 tensor"""
    image = read_file(image_file)
    image = decode_jpeg(image)

    return tf.cast(image, tf.float32)


def resize(image, height, width):
    image = tf.image.resize(image,
                            [height, width],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image


def normalize(image):
    return (image / 127.5) - 1


def load_image(image_file):
    image = load(image_file)
    image = resize(image, 384, 384)
    image = normalize(image)
    return image


bw_dataset = tf.data.Dataset.list_files(str(BW_PATH / '*.jpg'), shuffle=False)
bw_dataset = bw_dataset.map(
    load_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
bw_dataset = bw_dataset.batch(BATCH_SIZE)

color_dataset = tf.data.Dataset.list_files(
    str(COLOR_PATH / '*.jpg'), shuffle=False)
color_dataset = color_dataset.map(
    load_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
color_dataset = color_dataset.batch(BATCH_SIZE)

combined_dataset = tf.data.Dataset.zip((bw_dataset, color_dataset))


def encoder_block(filters, kernel_size, batchnorm=True):
    """Define an encoder block."""
    initializer = RandomNormal(0., 0.02, seed=42)

    result = Sequential()
    result.add(Conv2D(filters,
                      kernel_size,
                      strides=2,
                      padding='same',
                      kernel_initializer=initializer,
                      use_bias=False
                      )
               )

    if batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result


def decoder_block(filters, kernel_size, dropout=False):
    """Define a decoder block."""
    initializer = RandomNormal(0., 0.02, seed=42)

    result = Sequential()
    result.add(Conv2DTranspose(filters, kernel_size,
                               strides=2,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False)
               )

    result.add(BatchNormalization())

    if dropout:
        result.add(Dropout(0.5))

    result.add(ReLU())

    return result


def Discriminator():
    """Define the discriminator model."""
    initializer = RandomNormal(0., 0.02, seed=42)

    input = Input(shape=[384, 384, 3], name='input_image')
    target = Input(shape=[384, 384, 3], name='target_image')

    # (batch_size, 384, 384, channels*2)
    x = Concatenate()([input, target])

    down1 = encoder_block(64, 4, False)(x)  # (batch_size, 192, 192, 64)
    down2 = encoder_block(128, 4)(down1)  # (batch_size, 96, 96, 128)
    down3 = encoder_block(256, 4)(down2)  # (batch_size, 48, 48, 256)
    down4 = encoder_block(512, 4)(down3)  # (batch_size, 24, 24, 512)

    zero_pad1 = ZeroPadding2D()(down4)  # (batch_size, 26, 26, 512)
    conv = Conv2D(512, 4,
                  strides=(1, 1),
                  kernel_initializer=initializer,
                  use_bias=False)(zero_pad1)  # (batch_size, 25, 25, 512)

    batchnorm1 = BatchNormalization()(conv)

    leaky_relu = LeakyReLU()(batchnorm1)

    zero_pad2 = ZeroPadding2D()(leaky_relu)  # (batch_size, 26, 26, 512)

    last = Conv2D(1, 4,
                  strides=(1, 1),
                  kernel_initializer=initializer
                  )(zero_pad2)  # (batch_size, 25, 25, 1)

    return Model(inputs=[input, target], outputs=last)


def Generator():
    """Define the generator model."""
    inputs = Input(shape=[384, 384, 3])

    down_stack = [
        encoder_block(64, 4, batchnorm=False),  # (batch_size, 192, 192, 64)
        encoder_block(128, 4),  # (batch_size, 96, 96, 128)
        encoder_block(256, 4),  # (batch_size, 48, 48, 256)
        encoder_block(512, 4),  # (batch_size, 24, 24, 512)
        encoder_block(512, 4),  # (batch_size, 12, 12, 512)
        encoder_block(512, 4),  # (batch_size, 6, 6, 512)
        encoder_block(512, 4),  # (batch_size, 3, 3, 512)
    ]

    up_stack = [
        decoder_block(512, 4, dropout=True),  # (batch_size, 6, 6, 1024)
        decoder_block(512, 4, dropout=True),  # (batch_size, 12, 12, 1024)
        decoder_block(512, 4, dropout=True),  # (batch_size, 24, 24, 1024)
        decoder_block(512, 4),  # (batch_size, 48, 48, 1024)
        decoder_block(256, 4),  # (batch_size, 96, 96, 512)
        decoder_block(128, 4),  # (batch_size, 192, 192, 256)
        decoder_block(64, 4),  # (batch_size, 384, 384, 128)
    ]

    initializer = RandomNormal(0., 0.02, seed=42)
    last = Conv2DTranspose(filters=3,
                           kernel_size=4,
                           strides=2,
                           padding='same',
                           kernel_initializer=initializer,
                           activation='tanh')  # (batch_size, 384, 384, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    x = last(x)

    return Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(
        disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(
        disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    # plt.show()
    # plt.savefig(f"pred-{time.time()}.jpg")


generator = Generator()
discriminator = Discriminator()

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

log_dir = "logs/"
summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator(
            [input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
            disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                     discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
        tf.summary.scalar('disc_loss', disc_loss, step=step//1000)


def fit(dataset, steps):
    start = time.time()

    for step, (input_image, target) in dataset.repeat().take(steps).enumerate():
        if (step) % 1000 == 0:
            if step != 0:
                print(
                    f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

            start = time.time()

            # generate_images(generator, input_image, target)
            # print(f"Step: {step//1000}k")

        train_step(input_image, target, step)

        # Training step
        if (step+1) % 50 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 5000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


# fit(combined_dataset, steps=20001)
checkpoint.restore("./training_checkpoints/ckpt-3")

for bw, color in combined_dataset.take(10):
    generate_images(generator, bw, color)
