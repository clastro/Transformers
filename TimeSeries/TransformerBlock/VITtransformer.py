import tensorflow as tf

class ViTEmbeddings(tf.keras.layers.Layer):
    def __init__(self, patch_size, hidden_size, dropout=0.0, **kwargs):
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.patch_embeddings = tf.keras.layers.Conv1D(filters = hidden_size, kernel_size = patch_size, strides = patch_size)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def build(self, input_shape):
        self.cls_token = self.add_weight(shape=(1, 1, self.hidden_size), trainable = True, name = "cls_token")

        num_patches = input_shape[1] // self.patch_size
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 1, self.hidden_size), trainable = True, name ="position_embeddings"
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        input_shape = tf.shape(inputs)
        embeddings = self.patch_embeddings(inputs, training= training)

        cls_tokens = tf.repeat(self.cls_token, repeats = input_shape[0], axis=0)
        embeddings = tf.concat((cls_tokens,embeddings), axis=1)

        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings, training = training)

        return embeddings

class MLP(tf.keras.layers.Layer):
    def __init__(self, mlp_dim, out_dim=None, activation = "gelu", dropout =0.0, **kwargs):
        super().__init__(**kwargs)
        self.mlp_dim = mlp_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.dense1 = tf.keras.layers.Dense(self.mlp_dim)
        self.activation1 = tf.keras.layers.Activation(self.activation)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(input_shape[-1] if self.out_dim is None else self.out_dim)

    def call(self, inputs: tf.Tensor, training: bool =False):
        x = self.dense1(inputs)
        x = self.activation1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        return x

class Block(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        attention_dim,
        attention_bias,
        mlp_dim,
        attention_dropout = 0.0,
        activation = "gelu",
        dropout = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm_before = tf.keras.layers.LayerNormalization()
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads,
            attention_dim // num_heads,
            use_bias = attention_bias,
            dropout = attention_dropout,
        )
        self.norm_after = tf.keras.layers.LayerNormalization()
        self.mlp = MLP(mlp_dim = mlp_dim, activation = activation, dropout = dropout)

    def build(self, input_shape):
        super().build(input_shape)
        self.attn._build_from_signature(input_shape, input_shape)

    def call(self, inputs, training=False):
        x = self.norm_before(inputs, training=training)
        x = self.attn(x, x, training= training)
        x2 = self.norm_after(x, training=training)
        x2 = self.mlp(x2, training=training)
        return x + x2

    def get_attention_scores(self, inputs):
        x = self.norm_before(inputs, training=False)
        _, weights = self.attn(x, x, training=False, return_attention_scores = True)
        return weights

class VisionTransformer(tf.keras.Model):
    def __init__(
        self,
        patch_size,
        hidden_size,
        depth,
        num_heads,
        mlp_dim,
        dropout = 0.0,
        attention_bias = False,
        attention_dropout = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.embeddings = ViTEmbeddings(patch_size, hidden_size, dropout)
        self.blocks = [
            Block(
                num_heads,
                attention_dim = hidden_size,
                attention_bias = attention_bias,
                attention_dropout = attention_dropout,
                mlp_dim = mlp_dim,
                dropout = dropout,
            )
            for i in range(depth)
        ]

        self.norm = tf.keras.layers.LayerNormalization()
        self.head = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs: tf.Tensor, training: bool = False)-> tf.Tensor:
        x = self.embeddings(inputs, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        x = self.norm(x)
        x = x[:,0]
        return self.head(x)

    def get_last_selfattention(self, inputs: tf.Tensor):
        x = self.embeddings(inputs, training=False)
        for block in self.blocks[:-1]:
            x = block(x, training=False)
        return self.blocks[-1].get_attention_scores(x)


