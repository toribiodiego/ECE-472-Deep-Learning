import tensorflow as tf
import numpy as np
import yaml
import struct
from tqdm import tqdm



class ConfigManager:
    """
    Manage and retrieve configurations from YAML file.
    """
    _instance = None
    _config = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path

    def _load_config(self):
        if not self._config:
            with open(self.config_path, 'r') as stream:
                self._config = yaml.safe_load(stream)

    def get(self, section):
        if not self._config:
            self._load_config()
        return self._config.get(section, {})

    def get_learning_params(self):
        return self.get('learning')

    def get_model_params(self):
        return self.get('model')

    def get_architecture_params(self):
        return self.get('architecture')


class BaseLayer:
    """
    Provides utility methods for activations, weights, and biases.
    """

    @staticmethod
    def get_activation(activation_str):
        """
        Maps the provided activation string to its corresponding TensorFlow activation function.

        Returns:
            tf.Tensor: The corresponding TensorFlow activation function.
        """
        activation_map = {
            'relu': tf.nn.relu,
            'sigmoid': tf.nn.sigmoid,
            'tanh': tf.nn.tanh,
            'leaky_relu': tf.nn.leaky_relu,
            'softmax': tf.nn.softmax
        }
        return activation_map.get(activation_str, tf.nn.relu)

    @staticmethod
    def init_weights(shape, stddev=0.1):
        """
        Initializes weights using a truncated normal distribution.
        """
        return tf.Variable(tf.random.truncated_normal(shape, stddev=stddev))

    @staticmethod
    def init_biases(shape, value=0.1):
        """
        Initializes biases with a constant value.
        """
        return tf.Variable(tf.constant(value, shape=shape))


class DatasetLoader:
    """
    Method for loading, preprocessing, and splitting the dataset.
    """
    def __init__(self, config_manager):
        self.config_manager = config_manager

    def load_data(self, img_filename, lbl_filename):
        """
        Load dataset images and labels from given filenames.
        """
        # Load Images
        try:
            with open(img_filename, "rb") as img_file:
                magic, num, rows, cols = struct.unpack(">IIII", img_file.read(16))
                images = np.fromfile(img_file, dtype=np.uint8).reshape(num, rows, cols, 1)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset file {img_filename} not found in the current directory. Please ensure it's there."
            )
        # Load labels
        try:
            with open(lbl_filename, "rb") as lbl_file:
                _, num = struct.unpack(">II", lbl_file.read(8))
                labels = np.fromfile(lbl_file, dtype=np.uint8)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset file {lbl_filename} not found in the current directory. Please ensure it's there."
            )

        return images, labels

    def preprocess_data(self, images, labels):
        """
        Preprocess the dataset images and labels.
        """
        # Normalize images
        if images is not None:
            images = images.astype(np.float32) / 255.0
        # Convert labels to one-hot encoding
        if labels is not None:
            num_labels = len(set(labels))
            labels_onehot = np.zeros((labels.shape[0], num_labels))
            labels_onehot[np.arange(labels.shape[0]), labels] = 1
            labels = labels_onehot

        return images, labels

    def get_train_validation_data(self):
        """
        Load, preprocess, and split the training dataset into training and validation sets.
        """
        train_images, train_labels = self.load_data("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
        train_images, train_labels = self.preprocess_data(train_images, train_labels)

        validation_split = self.config_manager.get('training')["validation_split"]
        validation_size = int(train_images.shape[0] * validation_split)

        val_images = train_images[:validation_size]
        val_labels = train_labels[:validation_size]
        train_images = train_images[validation_size:]
        train_labels = train_labels[validation_size:]

        return train_images, train_labels, val_images, val_labels

    def get_test_data(self):
        """
        Load and preprocess the test dataset.
        """
        test_images, test_labels = self.load_data("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte")
        return self.preprocess_data(test_images, test_labels)


class Conv2d(BaseLayer):
    """
    A class for the Convolutional 2D layer of a neural network.
    """
    def __init__(self, layer_params):
        super().__init__()
        """
        Initialize the Conv2D layer with given layer parameters.
        """
        self.weights = self.init_weights([
            layer_params["filter_size"],
            layer_params["filter_size"],
            layer_params["input_depth"],
            layer_params["num_filters"]
        ])
        self.biases = self.init_biases([layer_params["num_filters"]])

        self.stride = layer_params.get("stride", 1)
        self.padding = layer_params.get("padding", "SAME")

        self.activation = self.get_activation(layer_params.get("activation", "relu"))

    def forward(self, input_data, training=True):
        """
        Forward pass through the Conv2D layer.

        Returns:
            tf.Tensor: The output tensor after the convolution operation and activation.
        """
        conv = tf.nn.conv2d(
            input_data,
            self.weights,
            strides=[1, self.stride, self.stride, 1],
            padding=self.padding
        )
        return self.activation(conv + self.biases)


class PoolingLayer:
    """
    A class for the pooling layer of a neural network.
    """
    VALID_POOLING_TYPES = ['max', 'avg']
    VALID_PADDING_TYPES = ['SAME', 'VALID']

    def __init__(self, layer_params):
        """
        Initialize the Pooling layer with given layer parameters.
        """
        self.pool_size = layer_params.get("pool_size", 2)
        self.stride = layer_params.get("stride", 2)
        self.pooling_type = layer_params.get("pooling_type", "max")
        self._validate_pooling_type()

        self.padding = layer_params.get("padding", "SAME")
        self._validate_padding_type()

        # Fetch default ksize and stride if not provided
        self.default_ksize = [1, self.pool_size, self.pool_size, 1]
        self.default_stride = [1, self.stride, self.stride, 1]

    def _validate_pooling_type(self):
        """
        Validate that the provided pooling type is supported.
        """
        if self.pooling_type not in self.VALID_POOLING_TYPES:
            raise ValueError(
                f"Invalid pooling type provided: {self.pooling_type}. "
                f"Supported types are: {self.VALID_POOLING_TYPES}"
            )

    def _validate_padding_type(self):
        """
        Validate that the provided padding type is supported.
        """
        if self.padding not in self.VALID_PADDING_TYPES:
            raise ValueError(
                f"Invalid padding type provided: {self.padding}. "
                f"Supported types are: {self.VALID_PADDING_TYPES}"
            )

    def forward(self, input_data, training=True):
        """
        Forward pass through the Pooling layer.

        Returns:
            tf.Tensor: The output tensor after the pooling operation.
        """
        if self.pooling_type == 'max':
            return tf.nn.max_pool(
                input_data,
                ksize=self.default_ksize,
                strides=self.default_stride,
                padding=self.padding
            )
        elif self.pooling_type == 'avg':
            return tf.nn.avg_pool(
                input_data,
                ksize=self.default_ksize,
                strides=self.default_stride,
                padding=self.padding
            )


class FullyConnectedLayer(BaseLayer):
    """
    A class representing for the fully connected layer of a neural network.
    """
    def __init__(self, input_neurons, output_neurons, activation="relu"):
        super().__init__()
        self.weights = self.init_weights([input_neurons, output_neurons])
        self.biases = self.init_biases([output_neurons])
        self.activation = self.get_activation(activation)

    def forward(self, input_data, training=True):
        """
        Forward pass through the Fully Connected layer.

        Returns:
            tf.Tensor: The output tensor after the fully connected operation.
        """
        fc = tf.matmul(input_data, self.weights)
        fc_with_biases = fc + tf.reshape(self.biases, [1, -1])
        return self.activation(fc_with_biases)


class DropoutLayer(BaseLayer):
    """
    A class for the dropout layer of a neural network.
    """
    def __init__(self, layer_params):
        super().__init__()
        self.dropout_rate = layer_params.get("dropout_rate", 0.5)

    def forward(self, input_data, training=True):
        """
        Forward pass through the Dropout layer.

        Returns:
            tf.Tensor: The output tensor after applying dropout (if training is True) or the original input tensor.
        """
        if training:
            return tf.nn.dropout(input_data, rate=self.dropout_rate)
        return input_data


class LayerFactory:
    """
    A factory class to create and return layer instances. 
    Creates instances of Conv2D, FullyConnected, Pooling, and Dropout layers.
    """

    @staticmethod
    def create_layer(layer_info):
        layer_type = layer_info.get("layer")

        if layer_type == "Conv2D":
            return Conv2d(layer_info)

        elif layer_type == "FullyConnected":
            return FullyConnectedLayer(
                input_neurons=layer_info["input_neurons"],
                output_neurons=layer_info["output_neurons"]
            )

        elif layer_type == "Pooling":
            return PoolingLayer(layer_info)

        elif layer_type == "Dropout":
            return DropoutLayer(layer_info)

        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")


class Classifier:
    """
    A neural network classifier class that constructs, trains, and evaluates a model.
    """
    def __init__(self, config_manager_or_architecture):
        # Check if a list (i.e., architecture) is passed
        if isinstance(config_manager_or_architecture, list):
            self.architecture_params = config_manager_or_architecture
        # Check if a config_manager object is passed
        else:
            self.config_manager = config_manager_or_architecture
            self.architecture_params = self.config_manager.get_architecture_params()

        self.trainable_vars = []
        self.layers = self._construct_layers()

    def _construct_layers(self):
        """
        Constructs and returns the neural network layers based on the architecture parameters.

        Returns:
            List[Any]: A list of initialized neural network layer objects.
        """
        layers = []
        for layer_info in self.architecture_params:
            layer = LayerFactory.create_layer(layer_info)
            layers.append(layer)

            if isinstance(layer, (Conv2d, FullyConnectedLayer)):
                self.trainable_vars.extend([layer.weights, layer.biases])

        return layers

    @property
    def trainable_variables(self):
        return self.trainable_vars

    def forward(self, input_data, training=True):
        """
        Performs a forward pass of the model.

        Returns:
            tf.Tensor: The output tensor after the forward pass.
        """
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer):
                input_data = tf.reshape(input_data, [input_data.shape[0], -1])
            input_data = layer.forward(input_data, training=training)
        return input_data

    def save(self, save_path):
        """
        Saves the model's trainable variables to a checkpoint.
        """
        try:
            checkpoint = tf.train.Checkpoint(variables=self.trainable_variables)
            checkpoint.write(save_path)
            print(f"Model saved successfully at {save_path}.")
        except Exception as e:
            print(f"Error saving model at {save_path}. Error: {str(e)}")

    def load(self, save_path):
        """
        Loads the model's trainable variables from a checkpoint.
        """
        try:
            checkpoint = tf.train.Checkpoint(variables=self.trainable_variables)
            checkpoint.restore(save_path)
            print(f"Model loaded successfully from {save_path}.")
        except Exception as e:
            print(f"Error loading model from {save_path}. Error: {str(e)}")


class Trainer:
    """
    Trainer class responsible for training a given neural network model,
    evaluating it, and saving the best model during training.
    """
    def __init__(self, model, config_manager, optimizer):
        self.model = model
        self.config_manager = config_manager
        self.optimizer = optimizer
        self.best_val_accuracy = 0.0

    def train_epoch(self, train_images, train_labels, batch_size):
        """
        Trains the model for one epoch and returns the average training loss.

        Returns:
            float: Average training loss for the epoch.
        """
        num_batches = len(train_images) // batch_size
        total_train_loss = 0.0

        for i in tqdm(range(num_batches), desc="Training"):
            batch_images = train_images[i * batch_size : (i + 1) * batch_size]
            batch_labels = train_labels[i * batch_size : (i + 1) * batch_size]

            with tf.GradientTape() as tape:
                model_output = self.model.forward(batch_images, training=True)
                base_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=model_output, labels=batch_labels
                    )
                )
                reg_loss = l2_regularization(
                    self.model.layers, self.config_manager.get_learning_params()["l2_lambda"]
                )
                total_loss = base_loss + reg_loss

            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

            total_train_loss += total_loss.numpy()

        avg_train_loss = total_train_loss / num_batches
        return avg_train_loss

    def validate(self, val_images, val_labels):
        """
        Validates the model on validation data and returns the accuracy.

        Returns:
            float: Validation accuracy.
        """
        val_output = self.model.forward(val_images, training=False)
        predicted_class = tf.argmax(val_output, axis=1)
        true_class = tf.argmax(val_labels, axis=1)
        correct_predictions = tf.equal(predicted_class, true_class)
        val_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return val_accuracy

    def save_model_if_best(self, val_accuracy, model_save_path):
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            try:
                self.model.save(model_save_path)
            except Exception as e:
                print(f"Error during model saving: {str(e)}")


def l2_regularization(layers, l2_lambda):
    """Compute the L2 regularization term for all weights in the model."""
    l2_loss = 0.0
    for layer in layers:
        if isinstance(layer, (Conv2d, FullyConnectedLayer)):
            l2_loss += tf.nn.l2_loss(layer.weights)
    return l2_lambda * l2_loss


def main():
    """
    Main function to set up, train, and evaluate the neural network model.
    """

    # 1. Configuration Initialization
    # Set up the configuration manager to handle hyperparameters and settings.
    config_manager = ConfigManager()

    # 2. Data Loading and Preprocessing
    # Load and preprocess the training, validation, and test datasets.
    data_loader = DatasetLoader(config_manager)
    (
        train_images,
        train_labels,
        val_images,
        val_labels,
    ) = data_loader.get_train_validation_data()
    test_images, test_labels = data_loader.get_test_data()

    # 3. Model Building
    # Construct the neural network classifier based on the architecture defined in the configuration.
    model = Classifier(config_manager)

    # 4. Setup Optimizer using TensorFlow's built-in Adam
    # Initialize the optimizer with the learning rate specified in the configuration.
    optimizer_config = config_manager.get_model_params()
    optimizer = tf.optimizers.Adam(learning_rate=optimizer_config["learning_rate"])

    batch_size = optimizer_config["batch_size"]
    model_save_path = config_manager.get("model_save")["path"]

    trainer = Trainer(model, config_manager, optimizer)

    # 5. Training Loop
    # Train the model for the specified number of epochs.
    for epoch in range(optimizer_config["epochs"]):
        avg_train_loss = trainer.train_epoch(train_images, train_labels, batch_size)
        val_accuracy = trainer.validate(val_images, val_labels)

        # Print training statistics for the current epoch.
        tqdm.write(
            f"Epoch {epoch+1}/{optimizer_config['epochs']}, "
            f"Training Loss: {avg_train_loss:.4f}, "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )

        # Save the model if it achieves better validation accuracy.
        trainer.save_model_if_best(val_accuracy, model_save_path)

    # 6. Evaluation
    # Load the best model and evaluate its performance on the test set.
    model.load(model_save_path)

    # Compute the test accuracy.
    test_output = model.forward(test_images, training=False)
    predicted_class = tf.argmax(test_output, axis=1)
    true_class = tf.argmax(test_labels, axis=1)
    correct_predictions = tf.equal(predicted_class, true_class)
    test_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    print(f"Test Accuracy: {test_accuracy:.4f}")


if __name__ == "__main__":
    main()