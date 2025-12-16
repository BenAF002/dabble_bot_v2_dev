###############################################
# GRU-based phoneme embedding to word decoder #
# kinda mid tbh                               #
###############################################

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt


class GRU(nn.Module):
    def __init__(self, input_size: int = (64+50), hidden_size: int = 64):
        super().__init__()

        # "Reset Gate" components -- in forward: r_t := σ(r_x(x) + r_h(h))
        self.rx = nn.Linear(input_size, hidden_size, bias=True)
        self.rh = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # "Update Gate" components -- in forward: z_t := σ(z_x(x) + z_h(h))
        self.zx = nn.Linear(input_size, hidden_size, bias=True)
        self.zh = nn.Linear(hidden_size, hidden_size, bias=True)

        # "Candidate Hidden State" components
        self.hx = nn.Linear(input_size, hidden_size, bias=True)
        self.hh = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, input, hidden = None):
        B = input.shape[0]
        if hidden is None:
            hidden = torch.zeros((B, self.hidden_size), device=input.device)

        rt = F.sigmoid(self.rx(input) + self.rh(hidden))         # reset gate
        zt = F.sigmoid(self.zx(input) + self.zh(hidden))         # update gate
        cand_ht = F.tanh(self.hx(input) + self.hh(rt * hidden))  # candidate hidden state

        ht = (1 - zt) * hidden + zt * cand_ht  # final hidden state
        return ht
    

class DecoderGRU(nn.Module):
    def __init__(
            self,
            input_size: int = (64 + 50), 
            hidden_size: int = 64, 
            # seq_len: int = 15,
            warm_start_embeddings=None,
        ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = 29  # vocab_size: 27 (a-z + padding) + 2 (SOS, EOS) = ~29

        # optionally inherit character embeddings from pretrained model
        if warm_start_embeddings is None:
            self.char_embedding = nn.Embedding(self.vocab_size, embedding_dim=64, padding_idx=0)
        else:
            self.char_embedding = warm_start_embeddings

        self.cell = GRU(input_size, hidden_size)
        self.proj = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, phonetic_input, target_seq, hidden=None):
        """Forward pass with teacher-forcing"""
        B, T = target_seq.shape
        if hidden is None:
            h_t = torch.zeros((B, self.hidden_size), device=phonetic_input.device)
        else:
            h_t = hidden
        
        char_emb = self.char_embedding(target_seq)
        phonetic_input = phonetic_input.unsqueeze(1).repeat(1, T, 1)
        x = torch.cat([char_emb, phonetic_input], dim=2)

        seq_outputs = []  # store outputs at each time step
        for t in range(T):
            x_t = x[:, t, :]           # index into time-step
            h_t = self.cell(x_t, h_t)  # recur through GRU
            output_t = self.proj(h_t)  # project to vocab_size

            seq_outputs.append(output_t)
                    
        logits = torch.stack(seq_outputs, dim=1)  # B, T, vs
        return logits

    def predict(self, phonetic_input, max_len: int = 17):
        """
        Generate predictions
        This method does not use forward because we need to recur over a different (generative) seq
        starting with the <sos> char (idx=27)
        """
        B = phonetic_input.shape[0]
        
        char_input = torch.full((B, 1), 27, dtype=torch.long, device=phonetic_input.device) # init with sos (27)
        h_t = torch.zeros((B, self.hidden_size), device=phonetic_input.device)  # init hidden state

        gen_seq = []
        finished_mask = torch.zeros(B, dtype=torch.bool, device=phonetic_input.device)
        
        with torch.no_grad():
            for t in range(max_len):
                char_emb = self.char_embedding(char_input).squeeze(1)
                x_t = torch.cat([char_emb, phonetic_input], dim=1)  # B, C := 64+50
                h_t = self.cell(x_t, h_t)  # B, H := hidden_size
                logits_t = self.proj(h_t)  # B, vocab_size := 29

                probs = F.softmax(logits_t, dim=-1)  # B, vocab_size
                next_char = torch.multinomial(probs, num_samples=1)  # B, 1
                gen_seq.append(next_char)

                # update finished mask
                finished_mask = finished_mask | (next_char == 28)  # boolean update for <eos> tokens

                # update char_input
                char_input = next_char

                # stop early if all batch samples have hit <eos>
                if finished_mask.all():
                    break

        output_seq = torch.cat(gen_seq, dim=1)
        return output_seq
    

def train_epoch(model, dataset, loss_fn, optimizer, batch_size, track_losses: bool = False):
    """Train for one epoch."""
    model.train()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    n_batches = len(dl) 
    total_loss = 0

    losses = []
    for yb, xb in dl:
        optimizer.zero_grad()
        
        # shift character sequences for training and loss
        input_seq = yb[:, :-1]  # ensures only CURRENT char is passed as input
        target_seq = yb[:, 1:]  # ensures only NEXT char is used as taget
        
        logits = model(xb, input_seq)
        
        logits = logits.view(-1, model.vocab_size)
        targets = target_seq.reshape(-1).to(dtype=torch.long)
        loss = loss_fn(logits, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if track_losses: losses.append(loss.item())

    return total_loss / n_batches, losses

@torch.no_grad()
def eval_loss(model, dataset, loss_fn, sample_frac=0.2):
    """
    Evaluate model loss on a random sample of data.
    """
    # Store and set evaluation mode
    was_training = model.training
    model.eval()

    X, Y = dataset.embeddings, dataset.encoded_words
    try:
        n_samples = max(1, int(X.shape[0] * sample_frac))
        indices = torch.randperm(X.shape[0], device=X.device)[:n_samples]
        
        xb, yb = X[indices], Y[indices]

        # shift character sequences for training and loss
        input_seq = yb[:, :-1]  # ensures only CURRENT char is passed as input
        target_seq = yb[:, 1:]  # ensures only NEXT char is used as taget

        logits = model.forward(xb, input_seq)     
        logits = logits.view(-1, model.vocab_size)
        targets = target_seq.reshape(-1).to(dtype=torch.long)
        loss = loss_fn(logits, targets)
        
        return loss.item()
    
    finally:
        # Always restore training state, even if error occurs
        if was_training:
            model.train()


def training_loop(model, train_data, test_data, loss_fn, optimizer, track_losses: bool = False,
               batch_size: int = 64, n_epochs: int = 10, eval_sample_frac: float = 0.4):
    """Full training loop over multiple epochs."""
    
    start = time.time()
    losses = []
    for epoch in range(n_epochs):
        train_loss, eloss = train_epoch(model, train_data, loss_fn, optimizer, batch_size, track_losses)
        test_loss = eval_loss(model, test_data, loss_fn, sample_frac=eval_sample_frac)
        run_time = time.time() - start
        eta = divmod((run_time / (epoch + 1)) * (n_epochs - epoch - 1), 60)
        print(
            f"Epoch {epoch+1}/{n_epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
            + " | ETA: {min:,.0f} min, {sec:,.0f} sec".format(min=eta[0], sec=eta[1])
        )
        if track_losses: losses.extend(eloss)
    if track_losses: 
        plt.plot(losses);