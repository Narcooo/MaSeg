import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import math
from dataset.Landsat30 import DataSet
class NeuralDecisionTree(keras.Model):
    def __init__(self, depth, num_features, used_features_rate, num_classes):
        super(NeuralDecisionTree, self).__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_classes = num_classes

        # Create a mask for the randomly selected features.
        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indicies = np.random.choice(
            np.arange(num_features), num_used_features, replace=False
        )
        self.used_features_mask = one_hot[sampled_feature_indicies]

        # Initialize the weights of the classes in leaves.
        self.pi = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=[self.num_leaves, self.num_classes]
            ),
            dtype="float32",
            trainable=True,
        )

        # Initialize the stochastic routing layer.
        self.decision_fn = layers.Dense(
            units=self.num_leaves, activation="sigmoid", name="decision"
        )

    def call(self, features):
        batch_size = tf.shape(features)[0]

        # Apply the feature mask to the input features.
        features = tf.matmul(
            features, self.used_features_mask, transpose_b=True
        )  # [batch_size, num_used_features]
        # Compute the routing probabilities.
        decisions = tf.expand_dims(
            self.decision_fn(features), axis=2
        )  # [batch_size, num_leaves, 1]
        # Concatenate the routing probabilities with their complements.
        decisions = layers.concatenate(
            [decisions, 1 - decisions], axis=2
        )  # [batch_size, num_leaves, 2]

        mu = tf.ones([batch_size, 1, 1])

        begin_idx = 1
        end_idx = 2
        # Traverse the tree in breadth-first order.
        for level in range(self.depth):
            mu = tf.reshape(mu, [batch_size, -1, 1])  # [batch_size, 2 ** level, 1]
            mu = tf.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[
                :, begin_idx:end_idx, :
            ]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = tf.reshape(mu, [batch_size, self.num_leaves])  # [batch_size, num_leaves]
        probabilities = keras.activations.softmax(self.pi)  # [num_leaves, num_classes]
        outputs = tf.matmul(mu, probabilities)  # [batch_size, num_classes]
        return outputs

class NeuralDecisionForest(keras.Model):
    def __init__(self, num_trees, depth, num_features, used_features_rate, num_classes):
        super(NeuralDecisionForest, self).__init__()
        self.ensemble = []
        # Initialize the ensemble by adding NeuralDecisionTree instances.
        # Each tree will have its own randomly selected input features to use.
        for _ in range(num_trees):
            self.ensemble.append(
                NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)
            )

    def call(self, inputs):
        # Initialize the outputs: a [batch_size, num_classes] matrix of zeros.
        batch_size = tf.shape(inputs)[0]
        outputs = tf.zeros([batch_size, num_classes])

        # Aggregate the outputs of trees in the ensemble.
        for tree in self.ensemble:
            outputs += tree(inputs)
        # Divide the outputs by the ensemble size to get the average.
        outputs /= len(self.ensemble)
        return outputs

learning_rate = 0.01
batch_size = 2650000
num_epochs = 10
hidden_units = [64, 64]
num_trees = 25
depth = 10
used_features_rate = 1.0
num_classes = 5

def run_experiment(model):

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    train_images_path = 'D:/Dataset/SemanticSegmentation/Landsat30/512_128/Train/'
    train_d = DataSet(data_root=train_images_path,
                               transforms=None)
    train_examples, train_labels = train_d()
    print("Start training the model...")
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))

    model.fit(train_dataset, epochs=num_epochs)
    print("Model training finished")

    print("Evaluating the model on the test data...")
    test_images_path = 'D:/Dataset/SemanticSegmentation/Landsat30/512_128/Test/'
    test_d = DataSet(data_root=test_images_path,
                               transforms=None)
    test_examples, test_labels = test_d()
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))

    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.float32
            )
        else:
            inputs[feature_name] = layers.Input(
                name=feature_name, shape=(), dtype=tf.string
            )
    return inputs
# def encode_inputs(inputs):
#     encoded_features = []
#     for feature_name in inputs:
#         if feature_name in CATEGORICAL_FEATURE_NAMES:
#             vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
#             # Create a lookup to convert a string values to an integer indices.
#             # Since we are not using a mask token, nor expecting any out of vocabulary
#             # (oov) token, we set mask_token to None and num_oov_indices to 0.
#             lookup = StringLookup(
#                 vocabulary=vocabulary, mask_token=None, num_oov_indices=0
#             )
#             # Convert the string input values into integer indices.
#             value_index = lookup(inputs[feature_name])
#             embedding_dims = int(math.sqrt(lookup.vocabulary_size()))
#             # Create an embedding layer with the specified dimensions.
#             embedding = layers.Embedding(
#                 input_dim=lookup.vocabulary_size(), output_dim=embedding_dims
#             )
#             # Convert the index values to embedding representations.
#             encoded_feature = embedding(value_index)
#         else:
#             # Use the numerical features as-is.
#             encoded_feature = inputs[feature_name]
#             if inputs[feature_name].shape[-1] is None:
#                 encoded_feature = tf.expand_dims(encoded_feature, -1)
#
#         encoded_features.append(encoded_feature)
#
#     encoded_features = layers.concatenate(encoded_features)
#     return encoded_features

def create_forest_model():
    inputs = create_model_inputs()
    # features = encode_inputs(inputs)
    features = layers.BatchNormalization()(inputs)
    num_features = features.shape[1]

    forest_model = NeuralDecisionForest(
        num_trees, depth, num_features, used_features_rate, num_classes
    )

    outputs = forest_model(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


forest_model = create_forest_model()

run_experiment(forest_model)