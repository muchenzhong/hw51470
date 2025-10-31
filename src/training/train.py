"""
train.py: Training utilities and loops for Transformer Language Model
TensorFlow implementation for mystery corpus training

Author: Eric Ewing
"""

import tensorflow as tf
import math
import os
import json
from typing import Tuple, Dict
import tqdm

# Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss."""
    return math.exp(loss)

def train(model, train_dataset, test_dataset, epochs=5, learning_rate=1e-3,
          wandb_run=None, checkpoint_dir="checkpoints", continue_training=False, submission_tracker=None) -> Tuple[tf.keras.Model, Dict[str, list]]:
    """
    Complete training function for language models.

    Args:
        model: Language model to train
        train_dataset: Training dataset
        test_dataset: Test dataset
        epochs: Number of epochs
        learning_rate: Learning rate
        wandb_run: Wandb run for logging
        tokenizer: Tokenizer for text generation
        checkpoint_dir: Directory to save checkpoints
        continue_training: Whether to continue training from latest checkpoint
        submission_tracker: Submission tracker for logging epoch results

    Returns:
        model: Trained model
    """
    # Ensure checkpoint directory exists otherwise create it
    os.makedirs(checkpoint_dir, exist_ok=True) 
    # TODO: Initialize your optimizer and loss function

    #optimizer = tf.keras.optimizers.Adam(learning_rate)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")

    #optimizer = NotImplementedError
    #loss_fn = NotImplementedError
    
    # TODO: Set up TensorFlow checkpointing with Checkpoint and CheckpointManager

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

    #checkpoint = NotImplementedError
    #checkpoint_manager = NotImplementedError
    
    # Handle checkpoint restoration for continue training
    start_epoch = 1
    if continue_training:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            # Extract epoch number from checkpoint name
            try:
                start_epoch = int(latest_checkpoint.split('-')[-1])
                print(f"Resuming from epoch {start_epoch}")
            except:
                print("Could not determine start epoch, starting from 0")
        else:
            print("No checkpoint found, starting fresh")
    
    # This is to keep track of model's performance during training
    history = {'train_loss': [], 'val_loss': [], 'perplexity': []}


    @tf.function
    def _train_step(batch_tokens: tf.Tensor):
        x = tf.cast(batch_tokens[:, :-1], tf.int32) 
        y = tf.cast(batch_tokens[:, 1:],  tf.int32)  
        with tf.GradientTape() as tape:
            logits = model(x, training=True)                 
            per_tok = loss_fn(y, logits)                      
            loss = tf.reduce_mean(per_tok)                    
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def _eval_step(batch_tokens: tf.Tensor):
        x = tf.cast(batch_tokens[:, :-1], tf.int32)
        y = tf.cast(batch_tokens[:, 1:],  tf.int32)
        logits = model(x, training=False)
        per_tok = loss_fn(y, logits)
        return tf.reduce_mean(per_tok)

    best_val = float("inf")

    # TODO: Train your model, keep track of metrics and log to wandb
    # NOTE: tqdm can be used to create progress bars for any iterable (epochs, batches, etc.)
    # You might find it useful to wrap your epoch loop with tqdm.tqdm(...) for visual feedback
    for current_epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs), desc="Training Progress", position=0):
        total_epochs = start_epoch + epochs - 1
        # TODO: Iterate over the training dataset and update model weights
        train_loss_sum, train_batches = 0.0, 0

        for i, batch in enumerate(train_dataset):
            loss = _train_step(batch)
            train_loss_sum += float(loss)
            train_batches += 1

            if wandb_run and (i % 100 == 0):
                wandb_run.log({"epoch": int(current_epoch), "train_loss_iter": float(loss)})

        train_loss_epoch = train_loss_sum / max(1, train_batches)

        # TODO: Iterate over the test dataset and compute validation loss
        # NOTE: Make sure to call reduce_mean on the loss
        val_loss_sum, val_batches = 0.0, 0
        for batch in test_dataset:
            vloss = _eval_step(batch)
            val_loss_sum += float(vloss)
            val_batches += 1

        val_loss_epoch = val_loss_sum / max(1, val_batches)

        # TODO: Calculate perplexity from validation loss
        # NOTE: Make sure to call reduce_mean on the loss
        perplexity = calculate_perplexity(val_loss_epoch)
        
        # TODO: Append metrics to history dictionary
        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['perplexity'].append(perplexity)
        

        # TODO: Log epoch metrics to the submission tracker (epoch, train_loss, val_loss, perplexity)
        if submission_tracker is not None:
            submission_tracker.log_epoch(int(current_epoch), float(train_loss_epoch), float(val_loss_epoch), float(perplexity))


        # TODO : Save model checkpoint periodically or if validation loss improves
        if val_loss_epoch < best_val:
            best_val = val_loss_epoch
            ckpt_path = checkpoint_manager.save(checkpoint_number=int(current_epoch))
            print(f"Saved checkpoint: {ckpt_path}")
        print(f"epoch {current_epoch:02d} | train {train_loss_epoch:.4f}  val {val_loss_epoch:.4f}  ppl {perplexity:.1f}")

        # Log metrics to wandb if available (recommended into batch loop for logging for better tracking)
        # NOTE: If using for batch, make sure to log epoch number, not batch number on some N interval
        if wandb_run:
            wandb_run.log({
                "epoch": int(current_epoch), # TODO: Current epoch number (one-index, so add 1
                "train_loss": float(train_loss_epoch),  # TODO: Calculate training loss
                "val_loss": float(val_loss_epoch),  # TODO: Calculate validation loss
                "perplexity": float(perplexity)  # TODO: Calculate perplexity
            })
        

    return model, history