import tensorflow as tf
from encoder.star_transformer import StarTransformer

# 1. Define hyperparameters
hidden_size = 512
num_layers = 6
num_head = 8
head_dim = 64
dropout = 0.1
max_len = 512
batch_size = 32
seq_len = 100

# 2. Instantiate the StarTransformer model
# This is how you call the constructor
model = StarTransformer(
    hidden_size=hidden_size,
    num_layers=num_layers,
    num_head=num_head,
    head_dim=head_dim,
    dropout=dropout,
    max_len=max_len
)

# 3. Create dummy input data
dummy_input = tf.random.uniform(shape=(batch_size, seq_len, hidden_size))
dummy_mask = tf.ones(shape=(batch_size, seq_len), dtype=tf.int32)

# 4. Pass the data through the model
output_nodes, output_relay = model(dummy_input, dummy_mask)

# 5. Print the output shapes
print("Output nodes shape:", output_nodes.shape)
print("Output relay shape:", output_relay.shape) 