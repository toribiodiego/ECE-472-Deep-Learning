import tensorflow as tf
import tensorflow_addons as tfa 
import os
import pytest
import numpy as np
from hw4 import Trainer, LayerFactory, GroupNorm, Conv2d, PoolingLayer, FullyConnectedLayer, DropoutLayer, ResidualBlock, Classifier
import os



def test_model_saving_loading():
    # Define the architecture based on the provided parameters
    architecture = [
        {
            "layer": "Conv2D",
            "filter_size": 5,
            "num_filters": 16,
            "input_depth": 3,
            "activation": "relu",
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
            "input_neurons": 3136,  # 16 * 14 * 14 for the previous pooling layer
            "output_neurons": 128,
            "activation": "relu",
        },
        {
            "layer": "Dropout",
            "dropout_rate": 0.5,
        },
        {
            "layer": "FullyConnected",
            "input_neurons": 128,
            "output_neurons": 10,
            "activation": "none",
        },
        {
            "layer": "ResidualBlock",
            "filters": 64,
            "kernel_size": 3,
            "groups": 32,
        },
    ]

    # Create the model and save the initial weights
    model = Classifier(architecture)
    initial_weights = [var.numpy() for var in model.trainable_variables]

    # Save and load the model
    model.save("temp_model.ckpt")
    model.load("temp_model.ckpt")

    # Compare the initial weights to the weights after loading
    loaded_weights = [var.numpy() for var in model.trainable_variables]
    for init_w, loaded_w in zip(initial_weights, loaded_weights):
        assert np.allclose(init_w, loaded_w), "Weights mismatch after loading."













