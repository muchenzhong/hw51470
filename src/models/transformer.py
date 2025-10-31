import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="transformer")
class AttentionMatrix(keras.layers.Layer):
    """Compute attention matrix"""

    def __init__(self, use_causal_mask=False, **kwargs):
        super().__init__(**kwargs)
        self.use_causal_mask = use_causal_mask

    def call(self, inputs):
        """
        Compute attention weights from K and Q matrices.

        Args:
            inputs: [K, Q] where K and Q are [batch_size, seq_length, embed_size]

        Returns:
            attention_weights: [batch_size, seq_length, seq_length]
        """
        K, Q = inputs

        # 1. Ensure consistent dtypes (cast to tf.float32)
        K = tf.cast(K, tf.float32)
        Q = tf.cast(Q, tf.float32)
        head_size = tf.cast(tf.shape(K)[-1], tf.float32)

        # TODO: Compute scaled dot-product attention scores and normalize
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(head_size)

        # TODO: If use_causal_mask is True, apply causal mask to prevent attending to future tokens
        # TODO: Apply softmax to get attention weights
        if self.use_causal_mask:
            T = tf.shape(Q)[1]
            causal = tf.linalg.band_part(tf.ones((T, T), dtype=tf.float32), -1, 0) 
            scores = scores + (1.0 - causal) * (-1e9)

        attention_weights = tf.nn.softmax(scores, axis=-1)  # [B, T, T]
        return attention_weights
        
        #return NotImplementedError

    def get_config(self):
        config = super().get_config()
        return config

@keras.saving.register_keras_serializable(package="transformer")
class AttentionHead(keras.layers.Layer):
    """Single attention head"""

    def __init__(self, input_size, output_size, use_causal_mask=False, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.use_causal_mask = use_causal_mask

        # TODO: Initialize linear projections for K, Q, V
        self.K_proj = keras.layers.Dense(self.output_size, use_bias=False)
        self.Q_proj = keras.layers.Dense(self.output_size, use_bias=False)
        self.V_proj = keras.layers.Dense(self.output_size, use_bias=False)
        # TODO: Initialize attention matrix computation (pass in use_causal_mask)
        self.attn = AttentionMatrix(use_causal_mask=self.use_causal_mask)

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        Apply single attention head.

        Args:
            inputs_for_keys: [batch_size, seq_length, input_size]
            inputs_for_values: [batch_size, seq_length, input_size]
            inputs_for_queries: [batch_size, seq_length, input_size]

        Returns:
            output: [batch_size, seq_length, output_size]
        """
        # 1. Ensure consistent dtypes
        inputs_for_keys = tf.cast(inputs_for_keys, tf.float32)
        inputs_for_values = tf.cast(inputs_for_values, tf.float32)
        inputs_for_queries = tf.cast(inputs_for_queries, tf.float32)

        # TODO: Apply linear transformations to get K, Q, V
        K = self.K_proj(inputs_for_keys)     
        Q = self.Q_proj(inputs_for_queries) 
        V = self.V_proj(inputs_for_values) 
        # TODO: Compute attention weights
        weights = self.attn([K, Q])
        # TODO: Apply attention to values
        output = tf.matmul(weights, V)

        #return NotImplementedError
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "output_size": self.output_size,
            "use_causal_mask": self.use_causal_mask
        })
        return config

@keras.saving.register_keras_serializable(package="transformer")
class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention mechanism"""

    def __init__(self, embed_size, num_heads=8, use_causal_mask=False, **kwargs):
        super().__init__(**kwargs)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.use_causal_mask = use_causal_mask

        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        # TODO: Create attention heads (pass in use_causal_mask)
        self.heads = [
            AttentionHead(self.embed_size, self.head_size, use_causal_mask=self.use_causal_mask)
            for _ in range(self.num_heads)
        ]
        # TODO: Initialize output projection (embed_size)
        self.out_proj = keras.layers.Dense(self.embed_size, use_bias=False)

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        Apply multi-head attention.

        Args:
            inputs_for_keys: [batch_size, seq_length, embed_size]
            inputs_for_values: [batch_size, seq_length, embed_size]
            inputs_for_queries: [batch_size, seq_length, embed_size]

        Returns:
            output: [batch_size, seq_length, embed_size]
        """
        # 1. Ensure consistent dtypes
        inputs_for_keys = tf.cast(inputs_for_keys, tf.float32)
        inputs_for_values = tf.cast(inputs_for_values, tf.float32)
        inputs_for_queries = tf.cast(inputs_for_queries, tf.float32)

        # TODO: Apply each attention head
        head_outputs = [
            h(inputs_for_keys, inputs_for_values, inputs_for_queries) for h in self.heads
        ] 
        # TODO: Concatenate head outputs
        concat = tf.concat(head_outputs, axis=-1)
        # TODO: Apply output projection
        return self.out_proj(concat)

        return NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "num_heads": self.num_heads,
            "use_causal_mask": self.use_causal_mask
        })
        return config
    
@keras.saving.register_keras_serializable(package="transformer")
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding for transformer inputs.
    Uses sinusoidal position encodings as described in "Attention Is All You Need".
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Create positional encoding matrix
        pe = self.get_positional_encoding(max_seq_length, d_model)
        self.positional_encoding = tf.Variable(
            initial_value=pe, trainable=False, name='positional_encoding'
        )

    def get_positional_encoding(self, seq_length: int, d_model: int) -> tf.Tensor:
        """Generate sinusoidal positional encodings.
        
        Here is a link to the tensorflow implementation we are following:
        https://www.tensorflow.org/text/tutorials/transformer#the_embedding_and_positional_encoding_layer
        """
        
        # This is a position matrix where each row is position 0,1,2,...seq_length-1
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]  # [seq_length, 1]

        # This is the division term for the angles (computed from the Attention is All You Need paper)
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))  # [d_model // 2]
        
        # This calculates the sine and cosine terms for even and odd indices respectively
        pe_sin = tf.sin(position * div_term)  # [seq_length, d_model // 2]
        pe_cos = tf.cos(position * div_term)  # [seq_length, d_model // 2]

        # Now we stack and reshape to get the final positional encoding matrix
        pe = tf.stack([pe_sin, pe_cos], axis=2)  # [seq_length, d_model // 2, 2]
        pe = tf.reshape(pe, [seq_length, d_model])  # [seq_length, d_model]

        # NOTE: We need to add a batch dimension for broadcasting later it to inputs        
        return tf.expand_dims(pe, axis=0)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            inputs with positional encodings added
        """
        seq_length = tf.shape(inputs)[1]

        # TODO: Extract appropriate slice of positional encodings
        # HINT: self.positional_encoding has shape [1, max_seq_length, d_model]
        pe = self.positional_encoding[:, :seq_length, :]  
        pe = tf.cast(pe, tf.float32)
        return tf.cast(inputs, tf.float32) + pe



    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_seq_length": self.max_seq_length
        })
        return config
    
@keras.saving.register_keras_serializable(package="transformer")
class LanguageTransformerBlock(keras.layers.Layer):
    """Single transformer block optimized for language modeling (no cross-attention)"""

    def __init__(self, embed_size, num_heads=8, ff_hidden_size=None, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.ff_hidden_size = ff_hidden_size or 4 * embed_size
        self.dropout_rate = dropout_rate
        self.use_causal_mask = True  # Always use causal mask for language modeling

        # TODO: Initialize self-attention
        self.mha = MultiHeadAttention(embed_size, num_heads, use_causal_mask=self.use_causal_mask)

        # TODO: Initialize feed-forward network (2 layers)
        
        # First layer: embed_size -> ff_hidden_size with activation
        # Second layer: ff_hidden_size -> embed_size
        self.ffn = keras.Sequential([
            keras.layers.Dense(self.ff_hidden_size, activation=tf.nn.gelu),
            keras.layers.Dense(self.embed_size),
        ])
        # TODO: Initialize layer normalization layers
        self.ln1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = keras.layers.LayerNormalization(epsilon=1e-5)
        # TODO: Initialize dropout layers
        self.dropout1 = keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=None):
        """
        Apply transformer block with residual connections and layer normalization.

        Args:
            inputs: [batch_size, seq_length, embed_size]
            training: Whether in training mode

        Returns:
            output: [batch_size, seq_length, embed_size]
        """
        # 1. Ensure consistent dtype
        inputs = tf.cast(inputs, tf.float32)

        # TODO: Self-attention with residual connection and layer norm
        x = inputs
        h = self.ln1(x)
        h = self.mha(h, h, h)                     
        h = self.dropout1(h, training=training)
        x = x + h           

        # TODO: Feed-forward with residual connection and layer norm
        h2 = self.ln2(x)
        h2 = self.ffn(h2)
        h2 = self.dropout2(h2, training=training)
        out = x + h2                           

        return out


    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "num_heads": self.num_heads,
            "ff_hidden_size": self.ff_hidden_size,
            "dropout_rate": self.dropout_rate
        })
        return config
    
@keras.saving.register_keras_serializable(package="transformer")
class TransformerLanguageModel(keras.Model):
    """
    Complete Transformer Language Model
    """

    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=None,
                 max_seq_length=512, dropout_rate=0.1, pad_token_id=0, **kwargs):
        super().__init__(**kwargs)

        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or 4 * d_model
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.pad_token_id = pad_token_id

        # TODO: Initialize token embeddings
        self.token_embedding = keras.layers.Embedding(self.vocab_size, self.d_model)
        # TODO: Create positional encodings (d_model, max_seq_length)
        self.pos_enc = PositionalEncoding(self.d_model, self.max_seq_length)
        # TODO: Initialize embedding dropout
        self.emb_dropout = keras.layers.Dropout(self.dropout_rate)
        # TODO: Create transformer blocks (n_layers of LanguageTransformerBlock)
        self.blocks = [
            LanguageTransformerBlock(self.d_model, self.n_heads, ff_hidden_size=self.d_ff, dropout_rate=self.dropout_rate)
            for _ in range(self.n_layers)
        ]
        # TODO: Initialize final layer normalization
        self.final_ln = keras.layers.LayerNormalization(epsilon=1e-5)
        # TODO: Initialize transformer dropout
        self.transformer_dropout = keras.layers.Dropout(self.dropout_rate)
        # TODO: Initialize output projection to vocabulary
        self.to_vocab = keras.layers.Dense(self.vocab_size, use_bias=False)

    def call(self, inputs, training=None):
        """
        Forward pass through the language model.

        Args:
            inputs: Token indices [batch_size, seq_length]
            training: Whether in training mode

        Returns:
            Logits over vocabulary [batch_size, seq_length, vocab_size]
        """
        # 1. Get token embeddings and scale by sqrt(d_model)
        embeddings = self.token_embedding(inputs)  # [batch_size, seq_length, d_model]
        embeddings = tf.cast(embeddings, tf.float32)
        embeddings = embeddings * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        

        # TODO: Add positional encodings (remember to slice to seq_length)
        x = self.pos_enc(embeddings)
        
        # TODO: Apply dropout to embeddings
        x = self.emb_dropout(x, training=training)
        
        # TODO: Pass embeddings through transformer blocks using a loop
        for blk in self.blocks:
            x = blk(x, training=training)

        # TODO: Apply final layer normalization
        x = self.final_ln(x)
        x = self.transformer_dropout(x, training=training)
        
        # TODO: Project to vocabulary and return logits
        logits = self.to_vocab(x)  # [B, T, vocab]
        return logits


    def get_config(self):
        """Get model configuration for saving."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'max_seq_length': self.max_seq_length,
            'dropout_rate': self.dropout_rate,
            'pad_token_id': self.pad_token_id
        }

def create_language_model(vocab_size: int, **kwargs) -> TransformerLanguageModel:
    """
    Factory function to create a language model with sensible defaults.

    Args:
        vocab_size: Size of the vocabulary
        **kwargs: Additional model parameters

    Returns:
        Initialized TransformerLanguageModel
    """
    default_config = {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_seq_length': 256,
        'dropout_rate': 0.1,
        'pad_token_id': 0
    }

    # Update with provided kwargs
    config = {**default_config, **kwargs}
    config['vocab_size'] = vocab_size

    return TransformerLanguageModel(**config)