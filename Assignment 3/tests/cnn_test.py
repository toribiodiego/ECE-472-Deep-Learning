import tensorflow as tf
import numpy as np
import pytest
import yaml
from cnn import Conv2DLayer, BaseLayer, DropoutLayer, FullyConnectedLayer, CNNModel, PoolingLayer




def test_conv2d_output_shape():
    conv2d_params = {
        "input_depth": 1,
        "num_filters": 32,
        "filter_size": 3,
        "stride": 1,
        "padding": "SAME",
    }

    batch_size = 16
    input_size = 28
    input_depth = conv2d_params["input_depth"]
    input_tensor = tf.random.normal([batch_size, input_size, input_size, input_depth])

    conv_layer = Conv2DLayer(conv2d_params)

    output = conv_layer.forward(input_tensor)

    stride = conv2d_params.get("stride", 1)
    filter_size = conv2d_params["filter_size"]
    num_filters = conv2d_params["num_filters"]
    padding = conv2d_params.get("padding", "SAME")

    if padding == "SAME":
        expected_height = input_size
        expected_width = input_size
    else:
        expected_height = (input_size - filter_size) // stride + 1
        expected_width = expected_height

    expected_output_shape = [batch_size, expected_height, expected_width, num_filters]

    assert output.shape == tuple(expected_output_shape), f"Expected shape {expected_output_shape}, but got {output.shape}."


def test_dropout():
    dropout_params = {"dropout_rate": 0.5}
    dropout_layer = DropoutLayer(dropout_params)

    input_data = tf.ones([10, 10])

    num_trials = 100
    avg_sum = 0.0
    for _ in range(num_trials):
        output_training = dropout_layer.forward(input_data, training=True)
        avg_sum += tf.reduce_sum(output_training).numpy()

    avg_sum /= num_trials

    tolerance = 10 
    lower_bound = 100 - tolerance
    upper_bound = 100 + tolerance

    assert lower_bound <= avg_sum <= upper_bound, f"Expected average sum between {lower_bound} and {upper_bound}, but got {avg_sum}."


def test_fc_output_shape():
    input_neurons = 784  # 28x28
    output_neurons = 128
    
    batch_size = 16
    input_tensor = tf.random.normal([batch_size, input_neurons])
    
    fc_layer = FullyConnectedLayer(input_neurons, output_neurons)
    output = fc_layer.forward(input_tensor)
    
    expected_output_shape = [batch_size, output_neurons]
    
    assert output.shape == tuple(expected_output_shape), f"Expected shape {expected_output_shape}, but got {output.shape}."


def test_model_forward_pass():
    architecture = [
        {
            "layer": "Conv2D",
            "input_depth": 1,
            "num_filters": 32,
            "filter_size": 3,
            "stride": 1,
            "padding": "SAME",
        },
        {
            "layer": "Pooling",
            "pool_size": 2,
            "stride": 2,
            "padding": "SAME",
            "pooling_type": "max",
        },
        {
            "layer": "FullyConnected",
            "input_neurons": 14*14*32,
            "output_neurons": 128
        }
    ]
    
    model = CNNModel(architecture)
    input_tensor = tf.random.normal([16, 28, 28, 1])
    output = model.forward(input_tensor)
    
    assert output.shape == (16, 128), f"Expected shape (16, 128), but got {output.shape}."


def test_pooling_output_shapcleae():
    pooling_params = {
        "pool_size": 2,
        "stride": 2,
        "padding": "SAME",
        "pooling_type": "max",
    }
    
    batch_size = 16
    input_size = 28
    input_depth = 1
    input_tensor = tf.random.normal([batch_size, input_size, input_size, input_depth])
    
    pooling_layer = PoolingLayer(pooling_params)
    output = pooling_layer.forward(input_tensor)
    
    stride = pooling_params.get("stride", 2)
    pool_size = pooling_params["pool_size"]
    padding = pooling_params.get("padding", "SAME")
    
    if padding == "SAME":
        expected_height = (input_size + stride - 1) // stride
        expected_width = expected_height
    else:
        expected_height = (input_size - pool_size) // stride + 1
        expected_width = expected_height

    expected_output_shape = [batch_size, expected_height, expected_width, input_depth]
    
    assert output.shape == tuple(expected_output_shape), f"Expected shape {expected_output_shape}, but got {output.shape}."


def test_model_saving_loading():

    architecture = [
    {
        "layer": "Conv2D",
        "input_depth": 1,
        "num_filters": 32,
        "filter_size": 3,
        "stride": 1,
        "padding": "SAME",
    },
    {
        "layer": "Pooling",
        "pool_size": 2,
        "stride": 2,
        "padding": "SAME",
        "pooling_type": "max",
    },
    {
        "layer": "FullyConnected",
        "input_neurons": 14*14*32,
        "output_neurons": 128
    }
    ]

    model = CNNModel(architecture)
    initial_weights = [var.numpy() for var in model.trainable_variables]
    
    model.save("temp_model.ckpt")
    model.load("temp_model.ckpt")
    
    loaded_weights = [var.numpy() for var in model.trainable_variables]
    
    for init_w, loaded_w in zip(initial_weights, loaded_weights):
        assert np.allclose(init_w, loaded_w), "Weights mismatch after loading."


