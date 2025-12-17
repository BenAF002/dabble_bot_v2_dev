import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import warnings

def train_epoch(model, dataset, optimizer, batch_size: int, track_losses: bool = False):
    """Train for one epoch."""
    model.train()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = len(dl) 
    total_loss = 0

    # determine model type
    model_type = model.__class__.__name__

    losses = []
    for xb, yb in dl:
        xb, yb = xb.to(model.device), yb.to(model.device)

        # swap these if decoding
        if model.model_name == 'DecoderGRU':
            xb, yb = yb, xb

        optimizer.zero_grad() 

        # cast to BF16 if supported
        if torch.cuda.is_bf16_supported():
            with torch.autocast(device_type=model.device, dtype=torch.bfloat16):
                logits, loss = model(xb, yb)
        else:
            logits, loss = model(xb, yb)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if track_losses: losses.append(loss.item())

    return total_loss / n_batches, losses

@torch.no_grad()
def eval_loss(model, dataset):
    """
    Evaluate model loss on a random sample of data.
    """
    # Store and set evaluation mode
    was_training = model.training
    model.eval()

    X, Y = dataset.encoded_words, dataset.embeddings
    X, Y = X.to(model.device), Y.to(model.device)
    # swap these if decoding
    if model.model_name == 'DecoderGRU':
        X, Y = Y, X
    try:
        indices = torch.randperm(X.shape[0], device=model.device)
        logits, loss = model(X[indices], Y[indices])
        return loss.item()
    
    finally:
        # Always restore training state, even if error occurs
        if was_training:
            model.train()

def training_loop(model, train_data, test_data, optimizer, 
                  batch_size: int = 64, n_epochs: int = 10):
    """Full training loop over multiple epochs."""
    
    # set torch matmul precision to high for better performance
    if torch.get_float32_matmul_precision() != 'high':
        torch.set_float32_matmul_precision('high')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        start = time.time()
        training_losses = []
        for epoch in range(n_epochs):
            train_loss, losses = train_epoch(model, train_data, optimizer, batch_size)
            test_loss = eval_loss(model, test_data)
            training_losses.append(losses)

            if model.device == 'cuda': torch.cuda.synchronize()
            run_time = time.time() - start
            eta = divmod((run_time / (epoch + 1)) * (n_epochs - epoch - 1), 60)
            print(
                f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
                + " | ETA: {min:,.0f} min, {sec:,.0f} sec".format(min=eta[0], sec=eta[1])
            )