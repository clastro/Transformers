class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, ff_dim, name="TransformerEncoderLayer", **kwargs):
        super(TransformerEncoderLayer, self).__init__(name=name, **kwargs)
        self.mha = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)

    def call(self, inputs, training):
        attn_output, _ = self.mha(inputs, inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, num_heads, ff_dim, name="TransformerDecoderLayer", **kwargs):
        super(TransformerDecoderLayer, self).__init__(name=name, **kwargs)
        self.mha1 = MultiHeadAttention(embedding_dim, num_heads)
        self.mha2 = MultiHeadAttention(embedding_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)
        self.dropout3 = tf.keras.layers.Dropout(0.1)

    def call(self, inputs, encoder_output, training):
        attn_output1, _ = self.mha1(inputs, inputs, inputs)
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(inputs + attn_output1)
        
        attn_output2, _ = self.mha2(out1, encoder_output, encoder_output)
        attn_output2 = self.dropout2(attn_output2, training=training)
        out2 = self.layernorm2(out1 + attn_output2)
        
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        
        return out3

encoder_layer = TransformerEncoderLayer(embedding_dim=64, num_heads=8, ff_dim=128)
decoder_layer = TransformerDecoderLayer(embedding_dim=64, num_heads=8, ff_dim=128)
