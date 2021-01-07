import tensorflow as tf
import numpy as np
from sklearn.metrics import pairwise_distances

tf.random.set_seed(3)
np.random.seed(3)

class MLPClustering(tf.keras.Model):
    def __init__(self, input_shape, n_cetroids=100):
        super(MLPClustering, self).__init__()
        n_samples = input_shape[0]
        fts = input_shape[-1]
        print(f'Number of samples: {n_samples}')
        print(f'Number of fts: {fts}')
        print(f'Number of clusters: {n_cetroids}')
        output_shape = fts*n_cetroids
        self.input_sample = tf.keras.layers.Input(shape=(fts,))
        self.bn = tf.keras.layers.BatchNormalization(batch_size=n_samples)
        self.fc1 = tf.keras.layers.Dense(units=10)
        self.fc2 = tf.keras.layers.Dense(units=10)
        self.fc3 = tf.keras.layers.Dense(units=output_shape)
        # for each sample we get 100 centroids of dimension fts
        self.reshape = tf.keras.layers.Reshape((n_cetroids, fts))

    def compile(self, loss, optimizer, metrics):
        # configure the loss, the optimizer and the metrics to evaluate the model
        self.compiled_loss = loss
        self.optimizer = optimizer
        self.compiled_metrics = metrics


    def __call__(self, x):

        #x = self.input_sample(x)
        x = self.bn(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.reshape(x)
        x = tf.reduce_mean(x, axis=0)
        return x


    def train_step(self, x):

        with tf.GradientTape() as tape:
            centroids = self(x)
            loss = self.compiled_loss(x, centroids, self.fc3.trainable_weights)

        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Return a dict mapping metric names to current value
        return {'loss': loss}



class ClusteringLoss(tf.keras.losses.Loss):
    def __init__(self):
        self.name = 'Clustering loss'

    def __call__(self, X, centroids, weights):

        # centroid has dimension (100, fts)
        # x has shape (n_sample, fts)
        loss = tf.constant(0, dtype=tf.float32)

        distance = tf.norm(X[:, None, :] - centroids[None, :, :], axis=-1)
        loss += (tf.reduce_sum(tf.reduce_min(distance, axis=1))) # this loss has shape (n_sample!/(n_sample-2)!n_sample! )

        predictions = tf.argmin(distance, axis=1)
        predictions = np.array(predictions)

        for cluster in np.unique(predictions):
            X_intra = X[predictions == cluster]
            distance = tf.norm(X_intra[:, None, :] - X_intra[None, :, :], axis=-1)
            loss += (tf.reduce_max(distance))

        for c_i in np.unique(predictions):
            for c_j in np.unique(predictions):
                if c_i > c_j:
                    X_i = X[predictions == c_i]
                    X_j = X[predictions == c_j]
                    distance =tf.norm(X_i[:, None, :] - X_j[None, :, :], axis=-1)
                    loss += (tf.reduce_min(distance))


        #loss += (tf.norm(weights[0]) + tf.norm(weights[1]))
        #print(tf.norm(weights[0])+ tf.norm(weights[1]))
        return loss




