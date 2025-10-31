import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="MyLayers")
#class VanillaRNN(tf.keras.layers.Model):
class VanillaRNN(tf.keras.Model):
    """
    Simple vanilla RNN implementation from scratch for text language modeling.
    Compatible with mystery dataset text modeling codebase.
    """

    def __init__(self, vocab_size, hidden_size, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_length = seq_length

        # TODO: Initialize embedding layer for token embeddings
        #self.embedding = None
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)
        
        # TODO: Initialize RNN weight matrices using self.add_weight from the Keras API
        self.Wxh = self.add_weight(
            name="Wxh", shape=(self.hidden_size, self.hidden_size),
            initializer="glorot_uniform", trainable=True
        )
        self.Whh = self.add_weight(
            name="Whh", shape=(self.hidden_size, self.hidden_size),
            initializer="glorot_uniform", trainable=True
        )
        self.bh = self.add_weight(
            name="bh", shape=(self.hidden_size,),
            initializer="zeros", trainable=True
        )
        # TODO: Initialize output projection layer to project to vocabulary size
        self.output_projection = tf.keras.layers.Dense(self.vocab_size, use_bias=True)
        #self.output_projection = None

    def call(self, inputs, training=None):
        """
        Forward pass for text language modeling.

        Args:
            inputs: Input token indices of shape [batch_size, seq_length]
            training: Training mode flag

        Returns:
            Logits of shape [batch_size, seq_length, vocab_size]
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]

        # 1. First, we pass the input tokens through the embedding layer
        embedded = self.embedding(inputs)  # [batch_size, seq_length, hidden_size]

        # 2. Now, we initalize the RNN hidden state to zeros and we will update it step by step
        h = tf.zeros([batch_size, self.hidden_size])

        # 3. Now, iterate through the sequence length (one "token" at a time) and compute RNN hidden states
        outputs = []
        for t in range(seq_length):
            x_t = embedded[:, t, :]  # [batch_size, hidden_size]
            # TODO: Compute RNN hidden state update using the RNN equation(s)
            h = tf.nn.tanh(tf.matmul(x_t, self.Wxh) + tf.matmul(h, self.Whh) + self.bh)
            outputs.append(h)

        # 4. Finally, we stack the outputs and project to vocabulary
        stacked_outputs = tf.stack(outputs, axis=1)  # [batch_size, seq_length, hidden_size]
        logits = self.output_projection(stacked_outputs)  # [batch_size, seq_length, vocab_size]

        return logits

    def get_config(self):
        base_config = super().get_config()
        config = {k:getattr(self, k) for k in ["vocab_size", "hidden_size", "seq_length"]}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

########################################################################################

@tf.keras.utils.register_keras_serializable(package="MyLayers")
#class LSTM(tf.keras.layers.Model):
class LSTM(tf.keras.Model): 
    """
    LSTM implementation for comparison with vanilla RNN.
    """

    def __init__(self, vocab_size, hidden_size, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_length = seq_length
        self.seq_length = seq_length

        # TODO: Initialize embedding layer for token embeddings
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.hidden_size)
        #self.embedding = None
        # TODO: Initialize LSTM weight matrices and biases using self.add_weight
        self.W = self.add_weight(
            name="W_concat",
            shape=(self.hidden_size + self.hidden_size, 4 * self.hidden_size),
            initializer="glorot_uniform",
            trainable=True,)

        def _forget_bias_init(shape, dtype=None):
            H = self.hidden_size
            b = tf.zeros(shape, dtype=dtype or tf.float32)
            return tf.concat([tf.ones((H,), dtype=b.dtype),tf.zeros((3*H,), dtype=b.dtype)], axis=0)

        self.b = self.add_weight(
            name="b_concat",
            shape=(4 * self.hidden_size,),
            initializer=_forget_bias_init,
            trainable=True,)
        
        # TODO: Initialize output layer to project to vocabulary size
        #self.output_projection = None
        self.output_projection = tf.keras.layers.Dense(self.vocab_size, use_bias=True)

    def call(self, inputs, training=None):
        """
        LSTM forward pass.

        Args:
            inputs: Input token indices [batch_size, seq_length]
            training: Training mode flag

        Returns:
            Logits [batch_size, seq_length, vocab_size]
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]

        # TODO: Embed input tokens
        x = self.embedding(inputs)

        # TODO: Initialize hidden and cell states
        h = tf.zeros([batch_size, self.hidden_size])
        c = tf.zeros([batch_size, self.hidden_size])

        outputs = []

        # TODO: Process sequence step by step with LSTM equations
        for t in range(seq_length):
            x_t = x[:, t, :]                   
            xh  = tf.concat([x_t, h], axis=-1)  
            gates = tf.matmul(xh, self.W) + self.b 
            f, i, o, g = tf.split(gates, 4, axis=-1)

            f = tf.nn.sigmoid(f)
            i = tf.nn.sigmoid(i)
            o = tf.nn.sigmoid(o)
            g = tf.nn.tanh(g)

            c = f * c + i * g
            h = o * tf.nn.tanh(c)

            outputs.append(h)
        

        # TODO: Stack outputs and compute logits of each output step
        H = tf.stack(outputs, axis=1)          
        logits = self.output_projection(H)    
        return logits

    def get_config(self):
        base_config = super().get_config()
        config = {k:getattr(self, k) for k in ["vocab_size", "hidden_size", "seq_length"]}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def create_rnn_language_model(vocab_size, hidden_size, seq_length, model_type="vanilla"):
    """
    Create an RNN-based language model.

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden state dimension
        seq_length: Maximum sequence length
        model_type: Type of RNN ("vanilla", "lstm")

    Returns:
        Configured RNN language model
    """
    # 1. Create appropriate RNN layer based on model_type
    if model_type == "vanilla":
        model = VanillaRNN(vocab_size, hidden_size, seq_length)
    elif model_type == "lstm":
        model = LSTM(vocab_size, hidden_size, seq_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'vanilla' or 'lstm'")

    return model