class DistanceAwareMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads=8, name="DistanceAwareMultiHeadAttention", **kwargs):
        super(DistanceAwareMultiHeadAttention, self).__init__(name=name, **kwargs)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0
        self.projection_dim = embedding_dim // num_heads

        self.query_dense = tf.keras.layers.Dense(embedding_dim)
        self.key_dense = tf.keras.layers.Dense(embedding_dim)
        self.value_dense = tf.keras.layers.Dense(embedding_dim)
        self.dense = tf.keras.layers.Dense(embedding_dim)

    def scaled_dot_product_attention(self, query, key, value, distance_matrix):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        depth = tf.cast(tf.shape(key)[-1], tf.float32)
        logits = matmul_qk / tf.math.sqrt(depth)

        # Adding distance-based penalty to the logits
        logits -= distance_matrix

        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # Calculate distance matrix
        distance_matrix = self.calculate_distance_matrix(seq_len)

        scaled_attention, scaled_attention_weights = self.scaled_dot_product_attention(query, key, value, distance_matrix)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, seq_len, self.embedding_dim))
        outputs = self.dense(concat_attention)
        return outputs, scaled_attention_weights

    def calculate_distance_matrix(self, seq_len):
        # Create a TensorFlow tensor for indices
        indices = tf.range(seq_len, dtype=tf.float32)
        distances = tf.abs(tf.expand_dims(indices, 1) - tf.expand_dims(indices,0))
        max_distance = tf.reduce_max(distances)
        normalized_distances = distances / max_distance
        return normalized_distances * 1e-02

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embedding_dim': self.embedding_dim,
            'num_heads': self.num_heads,
            "name": self.name
        })
        return config
