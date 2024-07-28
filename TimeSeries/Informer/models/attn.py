import tensorflow as tf
import numpy as np
from math import sqrt

class FullAttention(tf.keras.layers.Layer):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = tf.keras.layers.Dropout(attention_dropout)

    def call(self, queries, keys, values, attn_mask=None):
        B, L, H, E = tf.shape(queries)
        _, S, _, D = tf.shape(values)
        scale = self.scale or 1. / sqrt(float(E))

        scores = tf.einsum('blhe,bshe->bhls', queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = tf.linalg.band_part(tf.ones((L, L)), -1, 0)
            scores = tf.where(attn_mask, scores, -np.inf)

        A = self.dropout(tf.nn.softmax(scale * scores, axis=-1))
        V = tf.einsum('bhls,bshd->blhd', A, values)

        if self.output_attention:
            return V, A
        else:
            return V, None

class ProbAttention(tf.keras.layers.Layer):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = tf.keras.layers.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        B, H, L_K, E = tf.shape(K)
        _, _, L_Q, _ = tf.shape(Q)

        K_expand = tf.expand_dims(K, axis=-3)
        K_expand = tf.tile(K_expand, [1, 1, L_Q, 1, 1])
        index_sample = tf.random.uniform((L_Q, sample_k), minval=0, maxval=L_K, dtype=tf.int32)
        K_sample = tf.gather(K_expand, index_sample, batch_dims=-3)
        Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q, axis=-2), K_sample, transpose_b=True), axis=-2)

        M = tf.reduce_max(Q_K_sample, axis=-1) - tf.reduce_mean(Q_K_sample, axis=-1)
        M_top = tf.nn.top_k(M, k=n_top, sorted=False).indices

        Q_reduce = tf.gather(Q, M_top, batch_dims=-2)
        Q_K = tf.matmul(Q_reduce, K, transpose_b=True)

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = tf.shape(V)
        if not self.mask_flag:
            V_sum = tf.reduce_mean(V, axis=-2)
            context = tf.tile(tf.expand_dims(V_sum, axis=-2), [1, 1, L_Q, 1])
        else:
            context = tf.cumsum(V, axis=-2)
        return context

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = tf.shape(V)

        if self.mask_flag:
            attn_mask = tf.linalg.band_part(tf.ones((L_Q, L_V)), -1, 0)
            scores = tf.where(attn_mask, scores, -np.inf)

        attn = tf.nn.softmax(scores, axis=-1)

        context_in = tf.tensor_scatter_nd_update(context_in, index, tf.matmul(attn, V))
        if self.output_attention:
            attns = (tf.ones([B, H, L_V, L_V]) / L_V)
            attns = tf.tensor_scatter_nd_update(attns, index, attn)
            return context_in, attns
        else:
            return context_in, None

    def call(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = tf.shape(queries)
        _, L_K, _, _ = tf.shape(keys)

        queries = tf.transpose(queries, perm=[0, 2, 1, 3])
        keys = tf.transpose(keys, perm=[0, 2, 1, 3])
        values = tf.transpose(values, perm=[0, 2, 1, 3])

        U_part = self.factor * int(np.ceil(np.log(L_K)))
        u = self.factor * int(np.ceil(np.log(L_Q)))

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        scale = self.scale or 1. / sqrt(float(D))
        if scale is not None:
            scores_top *= scale

        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return tf.transpose(context, perm=[0, 2, 1, 3]), attn

class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = tf.keras.layers.Dense(d_keys * n_heads)
        self.key_projection = tf.keras.layers.Dense(d_keys * n_heads)
        self.value_projection = tf.keras.layers.Dense(d_values * n_heads)
        self.out_projection = tf.keras.layers.Dense(d_model)
        self.n_heads = n_heads
        self.mix = mix

    def call(self, queries, keys, values, attn_mask=None):
        B, L, _ = tf.shape(queries)
        _, S, _ = tf.shape(keys)
        H = self.n_heads

        queries = self.query_projection(queries)
        queries = tf.reshape(queries, (B, L, H, -1))
        keys = self.key_projection(keys)
        keys = tf.reshape(keys, (B, S, H, -1))
        values = self.value_projection(values)
        values = tf.reshape(values, (B, S, H, -1))

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        if self.mix:
            out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (B, L, -1))

        return self.out_projection(out), attn
