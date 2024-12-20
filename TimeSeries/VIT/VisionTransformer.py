import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
from einops.layers.tensorflow import Rearrange
from einops import rearrange, repeat

class MLP(Layer):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super(MLP, self).__init__()
        self.net = Sequential([
            nn.Dense(units=hidden_dim),
            nn.Activation(tf.keras.activations.gelu), 
            nn.Dropout(rate=dropout),
            nn.Dense(units=dim),
            nn.Dropout(rate=dropout)
        ])

    def call(self, x, training=True):
        return self.net(x, training=training)

class Attention(Layer):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.attend = nn.Softmax()
        self.to_qkv = nn.Dense(units=inner_dim * 3, use_bias=False)
        self.to_out = nn.Dense(units=dim) if not (heads == 1 and dim_head == dim) else None
        self.dropout = nn.Dropout(rate=dropout)

    def call(self, x, training=True):
        qkv = self.to_qkv(x)
        qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = tf.matmul(q, tf.transpose(k, perm=[0, 1, 3, 2])) * self.scale
        attn = self.attend(dots)

        x = tf.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        
        if self.to_out:
            x = self.to_out(x)
        x = self.dropout(x, training=training)

        return x

class ViT(Model):
    def __init__(self, seq_length, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', dim_head=64, dropout=0.0, emb_dropout=0.0):
        super(ViT, self).__init__()

        seq_len = seq_length
        patch_len = patch_size

        num_patches = seq_len // patch_len

        self.patch_embedding = Sequential([
            Rearrange('b (p l) c -> b (p) (l c)', p=patch_len),  
            nn.Dense(units=dim)
        ])

        self.pos_embedding = tf.Variable(initial_value=tf.random.normal([1, num_patches + 1, dim]), trainable=False)
        self.cls_token = tf.Variable(initial_value=tf.random.normal([1, 1, dim]), trainable=False)
        self.dropout = nn.Dropout(rate=emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool

        self.mlp_head = Sequential([
            nn.LayerNormalization(),
            nn.Dense(units=num_classes)
        ])

    def call(self, seq, training=True, **kwargs):
        x = self.patch_embedding(seq)
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x, training=training)

        x = self.transformer(x, training=training)

        if self.pool == 'mean':
            x = tf.reduce_mean(x, axis=1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x
