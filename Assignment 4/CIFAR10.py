import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pickle
import os
import yaml
import tarfile
from tqdm import tqdm
import json



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

    def __init__(self, config_path="10.yaml"):
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

    def get_data_augmentation_params(self):
        return self.get('data_augmentation')


class BaseLayer:
    """
    Provides utility methods for activations, weights, and biases.
    """
    @staticmethod
    def get_activation(activation_str, alpha=0.2, axis=-1):
        activation_map = {
            'relu': tf.nn.relu,
            'sigmoid': tf.nn.sigmoid,
            'tanh': tf.nn.tanh,
            'leaky_relu': lambda x: tf.nn.leaky_relu(x, alpha=alpha),
            'softmax': lambda x: tf.nn.softmax(x, axis=axis),
            'none': lambda x: x  # Added support for 'none' activation
        }
        
        if activation_str not in activation_map:
            supported_activations = ', '.join(activation_map.keys())
            raise ValueError(f"Unsupported activation function: {activation_str}. Supported activations: {supported_activations}")
        
        return activation_map[activation_str]

    @staticmethod
    def init_weights(shape, initializer='he_normal'):
        """
        Initializes weights using a specified distribution.
        
        Args:
            shape (list): Shape of the weight tensor.
            initializer (str): The initializer name as a string. Currently supports 'he_normal' and 'truncated_normal'.

        Returns:
            tf.Variable: The initialized weight variable.
        """
        if initializer == 'he_normal':
            return tf.Variable(tf.keras.initializers.HeNormal()(shape))
        elif initializer == 'truncated_normal':
            stddev = 0.1
            return tf.Variable(tf.random.truncated_normal(shape, stddev=stddev))
        else:
            raise ValueError(f"Initializer {initializer} not supported")


    @staticmethod
    def init_biases(shape, value=0.1, initializer='constant'):
        """
        Initializes biases with a specified value or distribution.
        
        Args:
            shape (list): Shape of the bias tensor.
            value (float): The value for biases if initializer is 'constant'.
            initializer (str): The initializer name as a string. Currently supports 'constant'.

        Returns:
            tf.Variable: The initialized bias variable.
        """
        if initializer == 'constant':
            return tf.Variable(tf.constant(value, shape=shape))
        else:
            raise ValueError(f"Initializer {initializer} not supported")


class DatasetLoader:
    def __init__(self, dataset_info):
        self.dataset_info = dataset_info
        self.data_augmenter = self._initialize_data_augmenter()
        
        if not self._check_extracted_files():
            self._download_and_extract()
        
    def _initialize_data_augmenter(self):
        aug_config = self.dataset_info.get('data_augmentation', {})
        return DataAugmenter(aug_config) if aug_config else None

    def _check_extracted_files(self):
        extract_path = self.dataset_info['extract_path']
        required_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
        return all(os.path.exists(os.path.join(extract_path, file)) for file in required_files)

    def _download_and_extract(self):
        # Check if the folder already exists
        if os.path.exists(os.path.join(os.getcwd(), "cifar-100-python")):
            return
            
        # Define the save_path where the tar file is expected to be found
        save_path = os.path.join(os.getcwd(), "cifar-100-python.tar.gz")
        
        # Define the extract_path where the tar file should be extracted
        extract_path = os.path.join(os.getcwd())
        
        # If the tar file exists, proceed with the extraction
        if os.path.exists(save_path):
            with tarfile.open(save_path, 'r:gz') as tar:
                tar.extractall(path=extract_path)
        else:
            raise FileNotFoundError(f"{save_path} not found in the current working directory. Please download it.")


    def _check_extracted_files(self):
        extract_path = os.path.join(os.getcwd(), "cifar-100-python")
        required_files = ['train', 'test']
        return all(os.path.exists(os.path.join(extract_path, file)) for file in required_files)


    def load_batch(self, batch_name):
        batch_file = os.path.join(os.getcwd(), "cifar-100-python", batch_name)
        if not os.path.exists(batch_file):
            raise FileNotFoundError(f"{batch_file} not found. Please check the dataset extraction and path.")
        
        with open(batch_file, 'rb') as fo:
            batch = pickle.load(fo, encoding='bytes')
        
        return batch
            
    def load_data(self, data_type):
        """
        Load the data: data_type is either 'train' or 'test'
        """
        file_path = os.path.join(self.dataset_info['extract_path'], 'cifar-100-python', data_type)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found. Ensure the dataset is extracted correctly.")
        
        with open(file_path, 'rb') as file:
            batch = pickle.load(file, encoding='bytes')
        
        images = batch[b'data'].reshape((-1, 3, 32, 32)).transpose((0, 2, 3, 1))
        labels = batch[b'fine_labels']
        return np.array(images), np.array(labels)

    
    # Normalization applied here for training data
    def get_train_validation_data(self):
        # Load training data
        train_images, train_labels = self.load_data("train")
        train_images = self.preprocess_and_normalize(train_images)
        train_images, train_labels = self.preprocess_data(train_images, train_labels)

        # Split the data into training and validation subsets
        validation_split = self.dataset_info.get('validation_split', 0.1)
        validation_size = int(train_images.shape[0] * validation_split)
        val_images = train_images[:validation_size]
        val_labels = train_labels[:validation_size]
        train_images = train_images[validation_size:]
        train_labels = train_labels[validation_size:]

        return train_images, train_labels, val_images, val_labels

    # Normalization applied here for test data
    def get_test_data(self):
        # Load test data
        test_images, test_labels = self.load_data("test")
        test_images = self.preprocess_and_normalize(test_images)
        return self.preprocess_data(test_images, test_labels)


    def preprocess_and_normalize(self, images):
        images = images.astype('float32')
        images /= 255.0
        return images
    
    def preprocess_data(self, images, labels):
        labels = np.array(labels)
        num_labels = len(set(labels))
        labels_onehot = np.zeros((labels.shape[0], num_labels))
        labels_onehot[np.arange(labels.shape[0]), labels] = 1
        return images, labels_onehot
    

class DataAugmenter:
    def __init__(self, config):
        self.config = config

    def random_crop(self, image):
        if self.config['random_crop']['enabled']:
            padding = self.config['random_crop'].get('padding', 0)
            image = tf.image.resize_with_crop_or_pad(image, 32 + 2*padding, 32 + 2*padding)
            image = tf.image.random_crop(image, size=[32, 32, 3])
        return image

    def random_rotation(self, image):
        if self.config['random_rotation']['enabled']:
            max_angle = self.config['random_rotation'].get('max_angle', 0)
            # Getting a random angle between -max_angle and max_angle
            angle = tf.random.uniform([], minval=-max_angle, maxval=max_angle, dtype=tf.float32)
            # Convert the angle to radians
            angle = angle * (np.pi / 180)
            image = tfa.image.rotate(image, angle)
        return image

    def random_horizontal_flip(self, image):
        if self.config['random_horizontal_flip']['enabled']:
            image = tf.image.random_flip_left_right(image)
        return image

    def color_jitter(self, image):
        if self.config['color_jitter']['enabled']:
            brightness = self.config['color_jitter'].get('brightness', 0)
            contrast = self.config['color_jitter'].get('contrast', 0)
            saturation = self.config['color_jitter'].get('saturation', 0)
            hue = self.config['color_jitter'].get('hue', 0)
            
            image = tf.image.random_brightness(image, max_delta=brightness)
            image = tf.image.random_contrast(image, lower=1-contrast, upper=1+contrast)
            image = tf.image.random_saturation(image, lower=1-saturation, upper=1+saturation)
            image = tf.image.random_hue(image, max_delta=hue)
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def cutout(self, image):
        if self.config['cutout']['enabled']:
            mask_size = self.config['cutout'].get('mask_size', 16)
            replace = self.config['cutout'].get('replace', 0)

            mask_value = image.dtype.type(replace)

            h, w, _ = image.shape
            x = tf.random.uniform([], 0, w, dtype=tf.int32)
            y = tf.random.uniform([], 0, h, dtype=tf.int32)

            mask = tf.ones([mask_size, mask_size], dtype=image.dtype) * mask_value

            mask_h_start = tf.maximum(0, y - mask_size // 2)
            mask_w_start = tf.maximum(0, x - mask_size // 2)
            mask_h_end = tf.minimum(h, mask_h_start + mask_size)
            mask_w_end = tf.minimum(w, mask_w_start + mask_size)

            mask_h_start = y - mask_size // 2
            mask_w_start = x - mask_size // 2
            mask_h_size = mask_h_end - mask_h_start
            mask_w_size = mask_w_end - mask_w_start

            padding_dims = [[mask_h_start, h - mask_h_end], [mask_w_start, w - mask_w_end]]
            mask = tf.image.pad_to_bounding_box([mask], *padding_dims, h, w)[0]

            mask = tf.broadcast_to(mask, [h, w, 3])
            image = tf.where(tf.equal(mask, mask_value), mask, image)
        return image

    @tf.function
    def augment(self, image):
        image = tf.convert_to_tensor(image)
        if self.config['random_crop']['enabled']:
            image = self.random_crop(image)
        if self.config['random_rotation']['enabled']:
            image = self.random_rotation(image)
        if self.config['random_horizontal_flip']['enabled']:
            image = self.random_horizontal_flip(image)
        if self.config['color_jitter']['enabled']:
            image = self.color_jitter(image)
            image = tf.clip_by_value(image, 0.0, 1.0)
        if self.config['cutout']['enabled']:
            image = self.cutout(image)
        if self.config['cutout']['enabled']:
            image = self.cutout(image)
        return image


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

    def __call__(self, input_data, training=True):
        conv = tf.nn.conv2d(
            input_data,
            self.weights,
            strides=[1, self.stride, self.stride, 1],
            padding=self.padding
        )
        return conv + self.biases


class PoolingLayer(BaseLayer):
    def __init__(self, layer_params):
        super().__init__()
        self.pool_size = layer_params.get("pool_size", 2)
        self.stride = layer_params.get("stride", 2)
        self.pooling_type = layer_params.get("pooling_type", "max")
        self.padding = layer_params.get("padding", "VALID")
        
    def __call__(self, input_data, training=True):
        if self.pooling_type == 'max':
            return tf.nn.max_pool2d(
                input_data,
                ksize=[1, self.pool_size, self.pool_size, 1],
                strides=[1, self.stride, self.stride, 1],
                padding=self.padding
            )
        elif self.pooling_type == 'avg':
            return tf.nn.avg_pool2d(
                input_data,
                ksize=[1, self.pool_size, self.pool_size, 1],
                strides=[1, self.stride, self.stride, 1],
                padding=self.padding
            )


class FullyConnectedLayer(BaseLayer):
    def __init__(self, input_neurons, output_neurons, activation="relu"):
        super().__init__()
        self.weights = self.init_weights([input_neurons, output_neurons])
        self.biases = tf.Variable(tf.zeros([output_neurons]), trainable=True)
        self.activation = self.get_activation(activation)

    def __call__(self, input_data, training=True):
        input_data = tf.reshape(input_data, [-1, self.weights.shape[0]])  # Modification here
        fc = tf.matmul(input_data, self.weights)
        fc_with_biases = fc + self.biases
        return self.activation(fc_with_biases)


class DropoutLayer(BaseLayer):
    """
    A class for the dropout layer of a neural network.
    """
    def __init__(self, layer_params):
        super().__init__()
        self.dropout_rate = layer_params.get("dropout_rate", 0.5)

    def __call__(self, input_data, training=True):
        """
        Invoke the Dropout layer.

        Returns:
            tf.Tensor: The output tensor after applying dropout (if training is True) or the original input tensor.
        """
        if training:
            return tf.nn.dropout(input_data, rate=self.dropout_rate)
        return input_data


class GroupNorm:
    def __init__(self, groups=32, epsilon=1e-5):
        self.desired_groups = groups
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None

    def __call__(self, inputs, training=True):
        if self.gamma is None:
            self.gamma = tf.Variable(tf.ones(inputs.shape[-1]), trainable=True)
            self.beta = tf.Variable(tf.zeros(inputs.shape[-1]), trainable=True)

        input_shape = tf.shape(inputs)
        N, H, W, C = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        
        # Adjust the number of groups
        G = self.desired_groups
        while C % G != 0:
            G -= 1
        
        x = tf.reshape(inputs, [N, G, H, W, C // G])
        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        x = tf.reshape(x, [N, H, W, C])
        return x * self.gamma + self.beta


class BatchNormalization:
    def __init__(self, momentum=0.99, epsilon=1e-5):
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_variance = None

    def __call__(self, inputs, training=True):
        if self.gamma is None:
            self.gamma = tf.Variable(tf.ones(inputs.shape[-1]), trainable=True)
            self.beta = tf.Variable(tf.zeros(inputs.shape[-1]), trainable=True)
            self.moving_mean = tf.Variable(tf.zeros(inputs.shape[-1]), trainable=False)
            self.moving_variance = tf.Variable(tf.ones(inputs.shape[-1]), trainable=False)

        reduction_axes = list(range(len(inputs.shape) - 1))

        def training_branch():
            mean, var = tf.nn.moments(inputs, reduction_axes, keepdims=True)
            mean = tf.squeeze(mean)
            var = tf.squeeze(var)

            # Update moving averages
            self.moving_mean.assign(self.moving_mean * self.momentum + mean * (1 - self.momentum))
            self.moving_variance.assign(self.moving_variance * self.momentum + var * (1 - self.momentum))
            return mean, var

        def inference_branch():
            mean = tf.reshape(self.moving_mean, [1] * len(reduction_axes) + [-1])
            var = tf.reshape(self.moving_variance, [1] * len(reduction_axes) + [-1])
            return mean, var

        # Convert the Python boolean `training` to a TensorFlow tensor
        training = tf.convert_to_tensor(training)

        # Use the converted `training` tensor in tf.cond
        mean, var = tf.cond(training, training_branch, inference_branch)

        x_normalized = (inputs - mean) / tf.sqrt(var + self.epsilon)
        return x_normalized * self.gamma + self.beta


class ResidualBlock(tf.Module):
    def __init__(self, layer_info):
        filters = layer_info["filters"]
        kernel_size = layer_info["kernel_size"]
        groups = min(layer_info.get("groups", filters), filters)
        
        # Ensure the number of groups is a divisor of the number of filters
        while filters % groups != 0:
            groups -= 1
        
        self.conv1 = Conv2d({
            "filter_size": kernel_size, 
            "input_depth": filters, 
            "num_filters": filters, 
            "activation": "relu"
        })
        self.groupnorm1 = GroupNorm(groups=groups)
        
        self.conv2 = Conv2d({
            "filter_size": kernel_size, 
            "input_depth": filters, 
            "num_filters": filters, 
            "activation": "relu"
        })
        self.groupnorm2 = GroupNorm(groups=groups)
        
        self.activation = tf.nn.relu
        
    def __call__(self, inputs, training=True):
        x = self.conv1(inputs, training=training)
        x = self.groupnorm1(x)
        x = self.activation(x)
        
        x = self.conv2(x, training=training)
        x = self.groupnorm2(x)
        
        x = x + inputs
        return self.activation(x)


class ActivationLayer(BaseLayer):
    """
    A class for an activation layer of a neural network.
    """
    def __init__(self, layer_params):
        super().__init__()
        self.activation = self.get_activation(layer_params.get("type", "relu"))

    def __call__(self, input_data, training=True):
        """
        Invoke the Activation layer.

        Returns:
            tf.Tensor: The output tensor after applying the activation function.
        """
        return self.activation(input_data)


class LayerFactory:
    """
    A class for an activation layer of a neural network.
    """
    def __init__(self, layer_params):
        super().__init__()
        self.activation = self.get_activation(layer_params.get("type", "relu"))

    def __call__(self, input_data, training=True):
        """
        Invoke the Activation layer.

        Returns:
            tf.Tensor: The output tensor after applying the activation function.
        """
        return self.activation(input_data)


class LayerFactory:
    @staticmethod
    def create_layer(layer_info):
        layer_type = layer_info.get("layer")

        if layer_type == "Conv2D":
            return Conv2d(layer_info)

        elif layer_type == "FullyConnected":
            return FullyConnectedLayer(
                input_neurons=layer_info["input_neurons"],
                output_neurons=layer_info["output_neurons"],
                activation=layer_info.get("activation", "relu")
            )

        elif layer_type == "Pooling":
            return PoolingLayer(layer_info)

        elif layer_type == "Dropout":
            return DropoutLayer(layer_info)
        
        elif layer_type == "ResidualBlock":
            return ResidualBlock(layer_info)
        
        elif layer_type == "BatchNormalization":
            return BatchNormalization()
        elif layer_type == "Activation":
            return ActivationLayer(layer_info)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")


class Classifier:
    def __init__(self, config_manager_or_architecture):
        if isinstance(config_manager_or_architecture, ConfigManager):
            self.config_manager = config_manager_or_architecture
            self.architecture_params = self.config_manager.get_architecture_params()
        else:
            # Assuming config_manager_or_architecture is directly the architecture parameters
            self.architecture_params = config_manager_or_architecture

        self.trainable_vars = []
        self.layers = self._construct_layers()

    def _construct_layers(self):
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

    def __call__(self, input_data, training=True):
        for layer in self.layers:
            input_data = layer(input_data, training=training)
        return input_data
    
    def save(self, save_path, optimizer):
        try:
            self.log_architectural_details(save_path)
            checkpoint = tf.train.Checkpoint(model=self, optimizer=optimizer)
            checkpoint.write(save_path)
            print(f"Model saved successfully at {save_path}.")
        except Exception as e:
            print(f"Error saving model at {save_path}. Error: {str(e)}")

    
    def load(self, save_path, optimizer):
        try:
            self.log_architectural_details(save_path)
            checkpoint = tf.train.Checkpoint(model=self, optimizer=optimizer)
            checkpoint.restore(save_path)
        except Exception as e:
            print(f"Error loading model from {save_path}. Error: {str(e)}")

    
    def log_architectural_details(self, save_path):
        try:
            # Constructing the path to save the architectural details
            dir_name = os.path.dirname(save_path)
            file_name = "architecture_details.json"
            architecture_path = os.path.join(dir_name, file_name)
            
            # Writing the architectural details to the file
            with open(architecture_path, 'w') as f:
                json.dump(self.architecture_params, f)

        except Exception as e:
            print(f"Error logging architectural details. Error: {str(e)}")


class Trainer:
    def __init__(self, model, config_manager, optimizer):
        self.model = model
        self.config_manager = config_manager
        self.optimizer = optimizer
        self.best_val_accuracy = 0.0

    def train_epoch(self, train_images, train_labels, batch_size):
        num_batches = len(train_images) // batch_size
        total_train_loss = 0.0

        for i in tqdm(range(num_batches), desc="Training"):
            batch_images = train_images[i * batch_size: (i + 1) * batch_size]
            batch_labels = train_labels[i * batch_size: (i + 1) * batch_size]

            with tf.GradientTape() as tape:
                model_output = self.model(batch_images, training=True)
                model_output = tf.reshape(model_output, [batch_size, -1])  # Reshaping the model output
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
        val_output = self.model(val_images, training=False)
        predicted_class = tf.argmax(val_output, axis=1)
        true_class = tf.argmax(val_labels, axis=1)
        correct_predictions = tf.equal(predicted_class, true_class)
        val_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return val_accuracy.numpy()

    def save_model_if_best(self, val_accuracy, model_save_path, optimizer):
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            try:
                self.model.save(model_save_path, optimizer)
            except Exception as e:
                print(f"Error during model saving: {str(e)}")


    def save_model_regularly(self, epoch, model_save_path):
        try:
            self.model.save(f"{model_save_path}_epoch_{epoch}")
        except Exception as e:
            print(f"Error during model checkpoint saving at epoch {epoch}: {str(e)}")

    def train_epoch(self, train_images, train_labels, batch_size):
        num_batches = len(train_images) // batch_size
        total_train_loss = tf.constant(0.0, dtype=tf.float32)

        for i in tqdm(range(num_batches), desc="Training"):
            batch_images = train_images[i * batch_size: (i + 1) * batch_size]
            batch_labels = train_labels[i * batch_size: (i + 1) * batch_size]
            total_train_loss += self.train_step(batch_images, batch_labels, batch_size)  # Passed batch_size as an argument
        avg_train_loss = total_train_loss / num_batches
        return avg_train_loss.numpy()  # Convert to numpy after the loop
    
    def validate_epoch(self, val_images, val_labels, batch_size):
        num_batches = len(val_images) // batch_size
        total_val_loss = 0.0
        total_correct_predictions = 0  # to accumulate the number of correct predictions
        
        for i in range(num_batches):
            batch_images = val_images[i * batch_size: (i + 1) * batch_size]
            batch_labels = val_labels[i * batch_size: (i + 1) * batch_size]
            
            model_output = self.model(batch_images, training=False)
            model_output = tf.reshape(model_output, [batch_size, -1])
            
            predicted_class = tf.argmax(model_output, axis=1)
            true_class = tf.argmax(batch_labels, axis=1)
            correct_predictions = tf.equal(predicted_class, true_class)
            total_correct_predictions += tf.reduce_sum(tf.cast(correct_predictions, tf.int32)).numpy()  # accumulate correct predictions
            
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=model_output, labels=batch_labels
                )
            )
            total_val_loss += loss.numpy()
        
        avg_val_loss = total_val_loss / num_batches
        val_accuracy = total_correct_predictions / (num_batches * batch_size)  # calculate validation accuracy
        return avg_val_loss, val_accuracy  # return validation accuracy along with average validation loss


def l2_regularization(layers, l2_lambda):
    """Compute the L2 regularization term for all weights in the model."""
    l2_loss = 0.0
    for layer in layers:
        if isinstance(layer, (Conv2d, FullyConnectedLayer)):
            l2_loss += tf.nn.l2_loss(layer.weights)
    return l2_lambda * l2_loss


def main():
    # 1. Configuration Initialization
    config_manager = ConfigManager()

    # Update the save_path to be in the current working directory
    model_save_filename = "best_model.ckpt"
    model_save_path = os.path.join(os.getcwd(), model_save_filename)

    # 2. Data Loading and Preprocessing
    dataset_info = config_manager.get('dataset_info')
    data_loader = DatasetLoader(dataset_info)
    train_images, train_labels, val_images, val_labels = data_loader.get_train_validation_data()
    test_images, test_labels = data_loader.get_test_data()

    # 3. Model Building
    architecture_info_key = 'architecture'
    model = Classifier(config_manager.get(architecture_info_key))

    # 4. Setup Optimizer
    model_config = config_manager.get('model')
    optimizer = tfa.optimizers.AdamW(learning_rate=model_config.get("learning_rate", 0.0001), 
                                     weight_decay=model_config.get("weight_decay", 0.01))

    # Get batch size from the configuration
    batch_size = model_config.get("batch_size", 64)

    # 5. Setup Trainer
    trainer = Trainer(model, config_manager, optimizer)
    patience = 10
    best_val_loss = float('inf')
    wait = 0

    # Determine starting epoch (modify this value if you want to continue training)
    starting_epoch = 0
    
    for epoch in range(starting_epoch, model_config.get("epochs", 50)):
        avg_train_loss = trainer.train_epoch(train_images, train_labels, batch_size)
        avg_val_loss, val_accuracy = trainer.validate_epoch(val_images, val_labels, batch_size)
        
        # Check if the validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0  # reset the counter
            trainer.save_model_if_best(val_accuracy, model_save_path, optimizer)
        else:
            wait += 1  # increment the counter if no improvement
            
        if wait >= patience:
            print("Early stopping due to no improvement")
            break
        
        # Computing training accuracy for the epoch
        train_output = model(train_images, training=False)
        predicted_train_class = tf.argmax(train_output, axis=1)
        true_train_class = tf.argmax(train_labels, axis=1)
        correct_train_predictions = tf.equal(predicted_train_class, true_train_class)
        train_accuracy = tf.reduce_mean(tf.cast(correct_train_predictions, tf.float32))
        
        # Reporting the parameter count along with other metrics
        parameter_count = np.sum([np.prod(var.shape) for var in model.trainable_variables])
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"Parameter Count: {parameter_count}")
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy.numpy():.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # 6. Final Evaluation
    print("Final Evaluating Model")
    model.load(model_save_path, optimizer)
    
    test_output = model(test_images, training=False)
    predicted_class = tf.argmax(test_output, axis=1)
    true_class = tf.argmax(test_labels, axis=1)
    correct_predictions = tf.equal(predicted_class, true_class)
    test_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    print(f"Final Test Accuracy: {test_accuracy.numpy():.4f}")






if __name__ == "__main__":
    main()
