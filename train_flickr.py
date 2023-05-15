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


PATH = pathlib.Path(__file__).parent / "MIRFLICKR" / "mirflickr"
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
    target = load(image_file)
    target = resize(target, 384, 384)
    target = normalize(target)

    bw = tf.image.rgb_to_grayscale(target)
    bw = tf.concat([bw, bw, bw], axis=2)
    return bw, target


def load_test_image(image_file):
    target = load(image_file)
    target = resize(target, 384, 384)

    bw = target
    if bw.shape[-1] == 3:
        bw = tf.image.rgb_to_grayscale(bw)

    bw = tf.concat([bw, bw, bw], axis=2)

    target = normalize(target)
    bw = normalize(bw)

    return bw, target


combined_dataset = tf.data.Dataset.list_files(str(PATH / "*.jpg"))
combined_dataset = combined_dataset.map(
    load_image, num_parallel_calls=tf.data.AUTOTUNE)
combined_dataset = combined_dataset.batch(BATCH_SIZE)


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

    display_list = [test_input[0], prediction[0]]
    title = ['Original', 'Upgraded']

    for i in range(2):
        plt.subplot(1, 2, i+1)
        plt.title(title[i])
        # Getting the pixel values in the [0, 1] range to plot.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')

    # plt.show()
    plt.savefig(f"pred-{time.time()}.jpg")
    plt.close()


generator = Generator()
discriminator = Discriminator()

checkpoint_dir = './flickr_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

log_dir = "flickr_logs/"
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
        if (step) % 100 == 0:
            if step != 0:
                print(
                    f'Time taken for 1000 steps: {time.time()-start:.2f} sec. Total steps: {step}.\n')

            start = time.time()

        train_step(input_image, target, step)

        # Training step
        if (step+1) % 10 == 0:
            print('.', end='', flush=True)

        # Save (checkpoint) the model every 10k steps
        if (step + 1) % 10000 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


checkpoint.restore("./flickr_checkpoints/ckpt-12")
# fit(combined_dataset, steps=101)

# for i, (bw, color) in enumerate(combined_dataset.take(50)):
#     if i % 10 == 5:
#         # generate_images(generator, bw, color)

TEST_DIR_BW = pathlib.Path(__file__).parent / "test_images" / "bw"
TEST_DIR_COLOR = pathlib.Path(__file__).parent / "test_images" / "color"
CHERRIES = pathlib.Path(__file__).parent / "test_images" / "cherries"


for path in os.listdir(CHERRIES):
    path = path.lower()
    if path.endswith('.jpg') or path.endswith('.jpeg'):
        bw, target = load_test_image(str(CHERRIES / path))

        bw = tf.expand_dims(bw, axis=0)
        target = tf.expand_dims(target, axis=0)

        generate_images(generator, bw, target)
