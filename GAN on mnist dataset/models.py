"""
BIOINFORMATICS: LAB07
@author: Irene Benedetto
"""

import tensorflow as tf

# create the discriminator model
class Discriminator(tf.keras.Model):
    """
        This class constitute simple implemtation of a Discriminator model in
        a Generative Adversarial Network. This model seeks to determine if a
        image is generated artificially or comes from the real dataset.

    """

    def __init__(self, input_shape, output_shape):
        super(Discriminator, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', input_shape=input_shape)
        self.relu1 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same')
        self.relu2 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()
        self.dense = tf.keras.layers.Dense(output_shape)

        self.compiled_loss = None
        self.optimizer = None
        self.compiled_metrics = None

    def compile(self, loss, optimizer, metrics):
        self.compiled_loss = loss
        self.optimizer = optimizer
        self.compiled_metrics = metrics
        
        self.model.summary()

    def __call__(self, x):
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.max_pool(x)
        x = self.dense(x)
        # x = self.prediction(x)
        return x

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x)
            # Compute the loss value
            loss = self.compiled_loss(y, y_pred)

        trainable_params = []
        for layer in [self.conv1, self.conv2, self.dense]:
            for param in layer.trainable_weights:
                trainable_params.append(param)

        gradients = tape.gradient(loss, trainable_params)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_params))
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {'loss': loss, 'accuracy': self.compiled_metrics.result().numpy()}


class Generator(tf.keras.Model):
    """
        This class constitute simple implementation of a Generator model in
        a Generative Adversarial Network. This model is trained to create new images
        starting from a random noise vector of dimension = latent_dim.
    """

    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        self.fc = tf.keras.layers.Dense(7 * 7 * 128, input_dim=latent_dim)
        self.relu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.reshape = tf.keras.layers.Reshape((7, 7, 128))

        self.conv1 = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")
        self.relu2 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv2 = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same")
        self.relu3 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.conv3 = tf.keras.layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid")

        self.compiled_loss = None
        self.optimizer = None
        self.compiled_metrics = None

    def compile(self, loss, optimizer, metrics):
        self.compiled_loss = loss
        self.optimizer = optimizer
        self.compiled_metrics = metrics
        
        self.model.summary()

    def __call__(self, x):

        x = self.fc(x)
        x = self.relu1(x)
        x = self.reshape(x)

        x = self.conv1(x)
        x = self.relu2(x)

        x = self.conv2(x)
        x = self.relu3(x)
        x = self.conv3(x)
        return x

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x)
            # Compute the loss value
            loss = self.compiled_loss(y, y_pred)

        trainable_params = []
        for layer in [self.fc, self.conv1, self.conv2, self.conv3]:
            for param in layer.trainable_weights:
                trainable_params.append(param)

        gradients = tape.gradient(loss, trainable_params)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_params))
        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {'loss': loss, 'accuracy': self.compiled_metrics.result().numpy()}


class GAN(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim, BATCH_SIZE):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim
        self.BATCH_SIZE = BATCH_SIZE

        self.compiled_loss = None
        self.d_optimizer = None
        self.g_optimizer = None
        self.gan_metrics = None

    def compile(self, loss, d_optimizer, g_optimizer, metrics):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss

    def train_step(self, real_data):
        (real_img, _) = real_data
        n_images = len(real_img)

        # label 1 means fake
        # label 0 means true

        # create random noise and fake images
        rand_noise = tf.random.normal(shape=(self.BATCH_SIZE, self.latent_dim))
        fake_img = self.generator(rand_noise)

        # Combine them with real images
        images = tf.concat([fake_img, real_img], axis=0)

        y = tf.concat([tf.ones((self.BATCH_SIZE, 1)), tf.zeros((n_images, 1))], axis=0)
        # Add random noise to the labels - important trick!
        y += 0.05 * tf.random.uniform(tf.shape(y))

        # Train the discriminator
        with tf.GradientTape() as tape:
            y_pred = self.discriminator(images)
            discriminator_loss = self.loss_fn(y, y_pred)

        grads = tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Sample again new random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(self.BATCH_SIZE, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((self.BATCH_SIZE, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator): it is trained to confuse!
        with tf.GradientTape() as tape:
            y_pred = self.discriminator(self.generator(random_latent_vectors))
            generator_loss = self.loss_fn(misleading_labels, y_pred)

        grads = tape.gradient(generator_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        return {"discriminator_loss": discriminator_loss, "generator_loss": generator_loss}

