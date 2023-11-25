import pytest
import tensorflow as tf
from hw6 import Linear, PositionalEncoding, FeedForwardNetwork, MultiHeadedAttention, TransformerModel, TransformerBlock


def test_linear():
    input_dim = 4
    output_dim = 2
    batch_size = 3
    seq_len = 5

    # Instantiate the Linear layer
    linear_layer = Linear(input_dim, output_dim)
    # Create a random tensor to simulate input data
    x = tf.random.uniform((batch_size, seq_len, input_dim), dtype=tf.float32)
    # Pass the input through the linear layer
    output = linear_layer(x)

    # Check if the output shape is as expected
    assert output.shape == (batch_size, seq_len, output_dim), "Incorrect output shape!"


def test_positional_encoding():
    d_model = 512
    max_len = 6000
    pos_encoding = PositionalEncoding(d_model, max_len)

    seq_len = 50
    pe = pos_encoding(seq_len)
    
    assert pe.shape == (seq_len, d_model), f"Positional encoding shape mismatch: expected ({seq_len}, {d_model}), got {pe.shape}"


def test_ffn():
    d_model = 4
    d_ff = 16
    dropout_rate = 0.1
    batch_size = 3
    seq_len = 5

    ffn_layer = FeedForwardNetwork(d_model, d_ff, dropout_rate)
    x = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)
    output = ffn_layer(x, training=True)

    assert output.shape == (batch_size, seq_len, d_model), "FFN layer incorrect output shape!"


def test_mha():
    d_model = 128
    num_heads = 8
    dropout_rate = 0.1
    batch_size = 3
    seq_len = 5
    mask = None  # Add a mask if needed, for example in sequence-to-sequence tasks

    mha_layer = MultiHeadedAttention(d_model, num_heads, dropout_rate)
    q = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)
    k = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)
    v = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)

    output = mha_layer(q, k, v, mask, training=False)
    
    assert output.shape == (batch_size, seq_len, d_model), "MHA test failed: incorrect output shape!"
    print("MHA test passed!")


def test_transformer_block():
    d_model = 128
    num_heads = 8
    d_ff = 512
    dropout_rate = 0.1
    batch_size = 3
    seq_len = 5
    mask = None  # Add a mask if needed, for example in sequence-to-sequence tasks

    transformer_block = TransformerBlock(d_model, num_heads, d_ff, dropout_rate)
    x = tf.random.uniform((batch_size, seq_len, d_model), dtype=tf.float32)

    output = transformer_block(x, mask, training=False)

    assert output.shape == (batch_size, seq_len, d_model), f"Transformer block test failed: expected output shape (batch_size, seq_len, d_model), got {output.shape}"
    print("Transformer block test passed!")


def test_transformer_model():
    vocab_size = 1000
    d_model = 128
    n_blocks = 4
    n_head = 8
    d_ff = 512
    dropout_rate = 0.1
    batch_size = 3
    seq_len = 5
    mask = None  # Add a mask if needed, for example in sequence-to-sequence tasks

    transformer_model = TransformerModel(vocab_size, d_model, n_blocks, n_head, d_ff, dropout_rate)
    inputs = tf.random.uniform((batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32)

    logits = transformer_model(inputs, mask, training=False)

    assert logits.shape == (batch_size, seq_len, vocab_size), f"Transformer model test failed: expected logits shape (batch_size, seq_len, vocab_size), got {logits.shape}"
    print("Transformer model test passed!")