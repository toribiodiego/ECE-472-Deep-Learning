import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from tqdm import tqdm
from linear import Linear


np.random.seed(23)
tf.random.set_seed(23)

class MLP(tf.Module):
    def __init__(
        self,
        num_inputs,
        num_outputs,
        num_hidden_layers,
        hidden_layer_width,
        hidden_activation,
        output_activation,
        l2_penalty=0.01,
        dropout_rate=0.0,
    ):
        # Store parameters as attributes
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.l2_penalty = l2_penalty
        self.dropout_rate = dropout_rate

        # Initialize layers
        self.layers = [Linear(num_inputs, hidden_layer_width)]
        for _ in range(num_hidden_layers - 1):
            self.layers.append(Linear(hidden_layer_width, hidden_layer_width))
        self.layers.append(Linear(hidden_layer_width, num_outputs))

        # Initialize weights and biases using custom initialization
        for layer in self.layers:
            layer.w.assign(self.custom_weight_initializer(layer.w.shape))
            if layer.bias:
                layer.b.assign(self.custom_bias_initializer(layer.b.shape))

        # Set activations, L2 penalty, and dropout rate
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def custom_weight_initializer(self, shape):
        n_in = shape[0]
        weights = tf.random.normal(shape, stddev=tf.math.sqrt(2.0 / n_in))
        return weights

    def custom_bias_initializer(self, shape):
        return tf.zeros(shape)

    def loss(self, predictions, labels):
        """
        Compute the binary cross-entropy loss.
        """
        epsilon = 1e-7  # Add a small epsilon for numerical stability
        loss_values = (
            -labels * tf.math.log(predictions + epsilon)
            - (1.0 - labels) * tf.math.log(1.0 - predictions + epsilon)
        )
        return tf.reduce_mean(loss_values)

    def reset_weights(self):
        """Reset the weights of the network using custom initializers."""
        for layer in self.layers:
            layer.w.assign(self.custom_weight_initializer(layer.w.shape))
            if layer.bias:
                layer.b.assign(self.custom_bias_initializer(layer.b.shape))

    def __call__(self, x, training=True):
        """Forward pass through the MLP."""
        for layer in self.layers[:-1]:
            x = self.hidden_activation(layer(x))
        return self.output_activation(self.layers[-1](x))

def grad_update(learning_rate, variables, gradients, clip_norm=1.0):
    clipped_grads, _ = tf.clip_by_global_norm(gradients, clip_norm)
    for v, grad in zip(variables, clipped_grads):
        v.assign_sub(learning_rate * grad)


def generate_data(n_points=250, noise=0.0, offset=0.35, plot=False):
    # Generate a single spiral
    theta = np.linspace(0, 3.5 * np.pi, n_points)
    radii = theta + offset
    x = radii * np.cos(-theta)
    y = radii * np.sin(-theta)

    rng = np.random.default_rng()
    noise_factor = 1 - np.exp(-radii / max(radii))
    x += noise * noise_factor * rng.standard_t(3, n_points)
    y += noise * noise_factor * rng.standard_t(3, n_points)

    # Blue spiral (previously red spiral)
    x1 = x + offset
    y1 = y

    # Red spiral (previously blue spiral)
    x2 = -x - offset
    y2 = -y

    blue_data = np.column_stack([x1, y1])
    red_data = np.column_stack([x2, y2])

    data = np.vstack([blue_data, red_data])
    labels = np.concatenate([np.ones(len(x1)), np.zeros(len(x2))])

    if plot:
        plot_spiral(x1, y1, x2, y2, color1="blue", color2="red")

    return data, labels


def train_model(mlp, data, labels, optimizer, learning_params):
    epoch_pbar = tqdm(
        range(learning_params["num_epochs"]), desc="Training Progress"
    )
    for epoch in epoch_pbar:
        batch_losses = []
        for i in range(0, len(data), learning_params["batch_size"]):
            x_batch = data[i : i + learning_params["batch_size"]]
            y_batch = labels[i : i + learning_params["batch_size"]]

            batch_loss = train_batch(mlp, x_batch, y_batch, optimizer)
            batch_losses.append(batch_loss)

        avg_batch_loss = np.mean(batch_losses)
        dataset_loss, dataset_accuracy = evaluate_model(mlp, data, labels)

        epoch_pbar.set_description(
            f"Epoch {epoch+1}/{learning_params['num_epochs']}, Batch Loss: {avg_batch_loss:.4f}, Overall Loss: {dataset_loss:.4f}, Accuracy: {dataset_accuracy:.4f}"
        )

    return mlp, avg_batch_loss, dataset_loss, dataset_accuracy


def plot_spiral(x1, y1, x2, y2, color1="red", color2="blue"):
    plt.scatter(
        x1,
        y1,
        color=color1,
        s=15,
        edgecolors="black",
        linewidth=0.75,
        label="Spiral 1",
    )
    plt.scatter(
        x2,
        y2,
        color=color2,
        s=15,
        edgecolors="black",
        linewidth=0.75,
        label="Spiral 2",
    )
    plt.title("Spiral")
    plt.axis("equal")
    plt.xlim(
        -max(abs(np.concatenate([x1, x2]))), max(abs(np.concatenate([x1, x2])))
    )
    plt.ylim(
        -max(abs(np.concatenate([y1, y2]))), max(abs(np.concatenate([y1, y2])))
    )
    plt.tight_layout()
    plt.legend()
    plt.show()


def compute_accuracy(predictions, targets):
    # Convert predictions to binary labels
    binary_predictions = tf.round(predictions)
    binary_predictions = tf.reshape(binary_predictions, [-1])  # Flatten the tensor

    correct_predictions = tf.equal(binary_predictions, targets)

    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy.numpy()


def plot_decision_boundary(model, X, y):
    """
    Plot the decision boundary of the model on data X, y.
    """
    # Define the grid
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict
    Z = model(tf.convert_to_tensor(np.c_[xx.ravel(), yy.ravel()], dtype=tf.float32))
    Z = Z.numpy().reshape(xx.shape)

    # Plot
    plt.contourf(xx, yy, Z, levels=[0.0, 0.5, 1.0], cmap="coolwarm", alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
    plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def setup_configuration_and_optimizer(config_path="config.yaml"):
    """
    Load configuration from a YAML file and set up the optimizer.

    Parameters:
    - config_path: Path to the YAML configuration file.

    Returns:
    - learning_params: Dictionary containing learning parameters.
    - model_params: Dictionary containing model parameters.
    - optimizer: Initialized Adam optimizer.
    """
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)
    
    learning_params = {
        'num_epochs': config['learning']['num_epochs'],
        'step_size': config['learning']['step_size'],
        'batch_size': config['learning']['batch_size']
    }
    
    model_params = {
        'num_inputs': config['model']['num_inputs'],
        'num_outputs': config['model']['num_outputs'],
        'num_hidden_layers': config['model']['num_hidden_layers'],
        'hidden_layer_width': config['model']['hidden_layer_width'],
        'l2_penalty': config['model']['l2_penalty'],
        'n_points': config['model']['n_points'], 
        'noise': config['model'].get('noise', 0.05), 
        'dropout_rate': config['model'].get('dropout_rate', 0.0), 
    }

    # Extract activation functions from config and map them
    hidden_activation_str = config['model'].get('hidden_activation', 'relu')
    output_activation_str = config['model'].get('output_activation', 'sigmoid')

    activation_map = {
        'relu': tf.nn.relu,
        'sigmoid': tf.nn.sigmoid,
        'tanh': tf.nn.tanh,
        'leaky_relu': tf.nn.leaky_relu,
        # Add more if needed
    }

    hidden_activation = activation_map[hidden_activation_str]
    output_activation = activation_map[output_activation_str]

    model_params['hidden_activation'] = hidden_activation
    model_params['output_activation'] = output_activation

    optimizer = tf.optimizers.Adam(learning_rate=learning_params['step_size'])
    
    return learning_params, model_params, optimizer


def initialize_mlp(model_params):
    print(f"Model Parameters: {model_params}")
    mlp = MLP(num_inputs=model_params['num_inputs'],
              num_outputs=model_params['num_outputs'],
              num_hidden_layers=model_params['num_hidden_layers'],
              hidden_layer_width=model_params['hidden_layer_width'],
              l2_penalty=model_params['l2_penalty'],
              hidden_activation=model_params['hidden_activation'],
              output_activation=model_params['output_activation'])
    return mlp


def train_batch(mlp, x_batch, y_batch, optimizer):
    """
    Train the model on a single batch.
    """
    with tf.GradientTape() as tape:
        predictions = mlp(x_batch)
        loss = mlp.loss(predictions, y_batch)

    grads = tape.gradient(loss, mlp.trainable_variables)
    optimizer.apply_gradients(zip(grads, mlp.trainable_variables))
    return loss.numpy()


def evaluate_model(mlp, val_x, val_y):
    """
    Evaluate the model on the validation set.
    """
    val_predictions = mlp(val_x)
    val_loss = mlp.loss(val_predictions, val_y)
    val_accuracy = compute_accuracy(val_predictions, val_y)
    return val_loss.numpy(), val_accuracy


def train_model(mlp, data, labels, optimizer, learning_params, file_writer=None):
    epoch_pbar = tqdm(
        range(learning_params["num_epochs"]), desc="Training Progress"
    )
    for epoch in epoch_pbar:
        batch_losses = []
        for i in range(0, len(data), learning_params["batch_size"]):
            x_batch = data[i : i + learning_params["batch_size"]]
            y_batch = labels[i : i + learning_params["batch_size"]]
            batch_loss = train_batch(mlp, x_batch, y_batch, optimizer)
            batch_losses.append(batch_loss)
        avg_batch_loss = np.mean(batch_losses)
        dataset_loss, dataset_accuracy = evaluate_model(mlp, data, labels)
        if file_writer:
            with file_writer.as_default():
                tf.summary.scalar(
                    "training_loss", avg_batch_loss, step=epoch * len(data)
                )
                tf.summary.scalar(
                    "dataset_loss", dataset_loss, step=epoch * len(data)
                )
        epoch_pbar.set_description(
            f"Epoch {epoch+1}/{learning_params['num_epochs']}, Batch Loss: {avg_batch_loss:.4f}, Overall Loss: {dataset_loss:.4f}, Accuracy: {dataset_accuracy:.4f}"
        )
    return mlp, avg_batch_loss, dataset_loss, dataset_accuracy



if __name__ == "__main__":
    learning_params, model_params, optimizer = setup_configuration_and_optimizer()
    mlp = initialize_mlp(model_params)
    mlp.reset_weights()
    data, labels = generate_data(
        model_params["n_points"],
        noise=model_params["noise"],
        offset=0.35,
        plot=False,
    )
    mlp, avg_batch_loss, dataset_loss, dataset_accuracy = train_model(
        mlp, data, labels, optimizer, learning_params
    )
    plot_decision_boundary(mlp, data, labels)