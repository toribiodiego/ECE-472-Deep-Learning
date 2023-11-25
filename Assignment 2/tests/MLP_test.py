import numpy as np
import pytest
import tensorflow as tf
import yaml
from linear import Linear
from MLP import MLP, initialize_mlp, setup_configuration_and_optimizer, train_model, generate_data




def test_load_yaml():
    learning_params, model_params, optimizer = setup_configuration_and_optimizer()

    # Load the configuration directly from the YAML file
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # Check if the loaded parameters match the expected values from the configuration
    assert learning_params["num_epochs"] == config["learning"]["num_epochs"]
    assert model_params["num_hidden_layers"] == config["model"]["num_hidden_layers"]


def test_optimizer_setup():
    _, _, optimizer = setup_configuration_and_optimizer()

    # Load the configuration directly from the YAML file
    with open("config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # Check if the optimizer is an instance of tf.optimizers.Adam and if its learning rate matches the expected value
    assert isinstance(optimizer, tf.optimizers.Adam)
    assert np.isclose(optimizer.lr.numpy(), config["learning"]["step_size"])


def test_training_data_shapes():
    data, labels = generate_data(n_points=250, noise=0.05, offset=0.35, plot=False)
    assert isinstance(data, np.ndarray) and isinstance(
        labels, np.ndarray
    ), "Data and labels should be numpy arrays."
    assert (
        data.shape[0] == labels.shape[0]
    ), "Number of data samples and labels should be the same."


def test_learning_parameters():
    learning_params, _, _ = setup_configuration_and_optimizer()
    assert learning_params["num_epochs"] > 0, "Number of epochs should be positive."
    assert learning_params["batch_size"] > 0, "Batch size should be positive."


def test_training_outputs():
    train_x = np.random.rand(100, 2)
    train_y = np.random.randint(0, 2, size=100)
    _, model_params, _ = setup_configuration_and_optimizer()
    mlp = initialize_mlp(model_params)
    optimizer = tf.optimizers.Adam()
    learning_params = {"num_epochs": 10, "batch_size": 10}
    _, avg_batch_loss, _, _ = train_model(
        mlp, train_x, train_y, optimizer, learning_params
    )

    assert isinstance(
        avg_batch_loss, (float, np.float32, np.float64)
    ), "Average batch loss should be a float."