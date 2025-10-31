"""
data.py - Data loading for Transformer Language Model
Simple loader for preprocessed mystery corpus pickle file
"""

import tensorflow as tf
import numpy as np
import pickle
from typing import List, Tuple, Dict
import re

class TextTokenizer:
    """
    Tokenizer wrapper for preprocessed vocabulary.
    """

    def __init__(self, vocab: List[str], word_to_idx: Dict[str, int]):
        """
        Initialize tokenizer with preprocessed vocabulary.

        Args:
            vocab: List of vocabulary tokens (i.e., words)
            word_to_idx: Dictionary mapping tokens to indices
        """
        self.vocab = vocab
        self.token_to_idx = word_to_idx
        self.idx_to_token = {idx: token for token, idx in word_to_idx.items()}

        # Special tokens (you'll get used to these the more you see language assignments)
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'

    def decode(self, indices: List[int]) -> str:
        """
        Convert token indices back to text.

        Args:
            indices: List of token indices

        Returns:
            Decoded text string
        """
        # TODO: Convert indices to tokens using self.idx_to_token
        mapped_tokens = [self.idx_to_token.get(int(i), self.unk_token) for i in indices]
        

        
        # TODO: Handle special tokens appropriately
        tokens = []
        for tok in mapped_tokens:
            if tok == self.pad_token or tok == self.eos_token:
                continue
            if tok == self.unk_token:
                tok = "unk"
            tokens.append(tok)
            
        # Join with spaces and fix punctuation spacing (provided for convenience and these are just for display)
        text = ' '.join(tokens)
        text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        text = text.replace(' :', ':').replace(' ;', ';').replace(' "', '"')

        return text

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token indices (basic implementation).

        Args:
            text: Input text to encode

        Returns:
            List of token indices
        """
        # Tokenize text (split on whitespace and punctuation; provided for convenience)
        # this returns a list of tokens you can iterate over (hint hint ;)
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        
        # TODO: Convert tokens to indices using self.token_to_idx
        raw_indices = [self.token_to_idx.get(tok) for tok in tokens]
        
        # TODO: Handle out-of-vocabulary words with unknown token
        indices = []
        unk_id = self.get_unk_token_id()
        indices = [idx if idx is not None else unk_id for idx in raw_indices]
        
        return indices

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def get_pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.token_to_idx.get(self.pad_token, 0)

    def get_unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.token_to_idx.get(self.unk_token, 1)

    def get_eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.token_to_idx.get(self.eos_token, 2)

    def __len__(self) -> int:
        return len(self.vocab)

def limit_vocabulary(train_tokens: List[int], test_tokens: List[int],
                    tokenizer: TextTokenizer, vocab_size: int) -> Tuple[List[int], List[int], TextTokenizer]:
    """
    Limit vocabulary size by keeping only the most frequent tokens.
    Remap all tokens to the reduced vocabulary.

    Args:
        train_tokens: Training token sequence
        test_tokens: Test token sequence
        tokenizer: Original tokenizer
        vocab_size: Target vocabulary size

    Returns:
        Tuple of (remapped_train_tokens, remapped_test_tokens, new_tokenizer)
    """
    # Keep the most frequent vocab_size tokens (they should already be sorted by frequency)
    # Ensure we keep special tokens
    special_tokens = ['<PAD>', '<UNK>', '<EOS>']

    # Keep top vocab_size tokens
    new_vocab = tokenizer.vocab[:vocab_size]

    # Make sure special tokens are included
    for special_token in special_tokens:
        if special_token not in new_vocab and special_token in tokenizer.vocab:
            # Replace least frequent non-special token
            for i in range(len(new_vocab)-1, -1, -1):
                if new_vocab[i] not in special_tokens:
                    new_vocab[i] = special_token
                    break

    # Create new word_to_idx mapping
    new_word_to_idx = {token: idx for idx, token in enumerate(new_vocab)}
    unk_idx = new_word_to_idx.get('<UNK>', 1)  # Default to index 1 if UNK not found

    # Remap tokens any token not in new vocab becomes UNK
    def remap_tokens(tokens):
        remapped = []
        for token_idx in tokens:
            original_token = tokenizer.idx_to_token.get(token_idx, '<UNK>')
            if original_token in new_word_to_idx:
                remapped.append(new_word_to_idx[original_token])
            else:
                remapped.append(unk_idx)
        return remapped

    new_train_tokens = remap_tokens(train_tokens)
    new_test_tokens = remap_tokens(test_tokens)

    # Create new tokenizer
    new_tokenizer = TextTokenizer(new_vocab, new_word_to_idx)

    print(f"  Vocabulary reduced from {len(tokenizer.vocab)} to {len(new_vocab)} tokens")
    print(f"  Special tokens: {[t for t in special_tokens if t in new_vocab]}")

    return new_train_tokens, new_test_tokens, new_tokenizer

def load_mystery_data(pickle_path: str = 'mystery_data.pkl') -> Tuple[List[int], List[int], TextTokenizer]:
    """
    Load preprocessed mystery corpus data from pickle file.

    Args:
        pickle_path: Path to the mystery_data.pkl file

    Returns:
        Tuple of (train_tokens, test_tokens, tokenizer)
    """
    print(f"Loading mystery data from {pickle_path}...")

    with open(pickle_path, 'rb') as f:
        data_dict = pickle.load(f)

    train_data = data_dict['train_data']
    test_data = data_dict['test_data']
    vocab = data_dict['vocab']
    word_to_idx = data_dict['word_to_idx']

    # Create tokenizer
    tokenizer = TextTokenizer(vocab, word_to_idx)

    return train_data, test_data, tokenizer

def create_sequences(tokens: List[int], seq_length: int) -> np.ndarray:
    """
    Create non-overlapping input sequences for language modeling.
    Targets will be computed during training by shifting the inputs.

    HINT: Your sequences should be of the length seq_length + 1 to accommodate target shifting.
    Args:
        tokens: List of token indices
        seq_length: Sequence length for training

    Returns:
        Input sequences array
    """
    sequences = []
    # TODO: Create non-overlapping sequences from token list

    step = seq_length
    for i in range(0, len(tokens) - seq_length, step):
        seq = tokens[i:i + seq_length + 1]
        if len(seq) == seq_length + 1:
            sequences.append(seq)
    return np.array(sequences, dtype=np.int32)



def create_tf_datasets(train_tokens: List[int], test_tokens: List[int],
                      seq_length: int = 256, batch_size: int = 16) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create TensorFlow datasets from token sequences.

    Args:
        train_tokens: Training token sequence
        test_tokens: Test token sequence
        seq_length: Sequence length for training
        batch_size: Batch size

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # TODO: Create sequences using create_sequences
    train_seq = create_sequences(train_tokens, seq_length)
    test_seq  = create_sequences(test_tokens,  seq_length)
    # TODO: Convert to TensorFlow constants (tf.constant will be your friend here)
    train_tensor = tf.constant(train_seq, dtype=tf.int32)
    test_tensor  = tf.constant(test_seq,  dtype=tf.int32)
    # TODO: Create datasets with batching and shuffling (look up the documentation for tf.data.Dataset)
    buf_size = int(train_tensor.shape[0]) if train_tensor.shape[0] is not None else 10000

    train_dataset = (tf.data.Dataset.from_tensor_slices(train_tensor)
                     .shuffle(buffer_size=buf_size)
                     .batch(batch_size, drop_remainder=True)   
                     .prefetch(tf.data.AUTOTUNE))

    test_dataset = (tf.data.Dataset.from_tensor_slices(test_tensor)
                    .batch(batch_size, drop_remainder=True)   
                    .prefetch(tf.data.AUTOTUNE))

    return train_dataset, test_dataset

def prepare_data(pickle_path: str = 'mystery_data.pkl', seq_length: int = 256,
                batch_size: int = 16, vocab_size: int = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, TextTokenizer]:
    """
    Complete data preparation pipeline for mystery corpus. You do not need to know the details of this function,
    but you should understand its inputs and outputs.

    Args:
        pickle_path: Path to mystery_data.pkl file
        seq_length: Sequence length for training
        batch_size: Batch size for datasets
        vocab_size: Maximum vocabulary size (None = use full vocab, smaller values = faster training)

    Returns:
        Tuple of (train_dataset, test_dataset, tokenizer)
    """
    print("=" * 60)
    print("PREPARING MYSTERY CORPUS DATA FOR BRUNO")
    print("=" * 60)

    # Load preprocessed data
    train_tokens, test_tokens, tokenizer = load_mystery_data(pickle_path)

    # Optionally limit vocabulary size for faster training
    if vocab_size is not None and vocab_size < len(tokenizer.vocab):
        print(f"Limiting vocabulary from {len(tokenizer.vocab)} to {vocab_size} tokens...")
        train_tokens, test_tokens, tokenizer = limit_vocabulary(
            train_tokens, test_tokens, tokenizer, vocab_size
        )

    # Create TensorFlow datasets
    train_dataset, test_dataset = create_tf_datasets(
        train_tokens, test_tokens, seq_length, batch_size
    )

    print("Data preparation complete!")
    print("=" * 60)

    return train_dataset, test_dataset, tokenizer

