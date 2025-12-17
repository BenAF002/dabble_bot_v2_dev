###############################################
# GRU-based phoneme embedding to word decoder #
# kinda mid but simple                        #
###############################################

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            warm_start_embeddings=None, 
        ):
        super().__init__()
        self.model_name = 'DecoderGRU'  # used for identification after compilation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = 32  # vocab_size: 27 (a-z + padding) + 2 (SOS, EOS) + nulls for dim alignment

        # TODO: optionally inherit character embeddings from pretrained model
        if warm_start_embeddings is None:
            self.char_embedding = nn.Embedding(self.vocab_size, embedding_dim=64, padding_idx=0)
        else:
            self.char_embedding = warm_start_embeddings

        self.cell = GRU(input_size, hidden_size)
        self.proj = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, phonetic_input, char_seq, hidden=None):
        """Forward pass with teacher-forcing"""
        input_seq = char_seq[:, :-1]  # ensures only CURRENT char is passed as input
        target_seq = char_seq[:, 1:]  # ensures only NEXT char is used as target
        B, T = input_seq.shape

        if hidden is None:
            h_t = torch.zeros((B, self.hidden_size), device=phonetic_input.device)
        else:
            h_t = hidden
        
        char_emb = self.char_embedding(input_seq)
        phonetic_input = phonetic_input.unsqueeze(1).repeat(1, T, 1)
        x = torch.cat([char_emb, phonetic_input], dim=2)

        seq_outputs = []  # store outputs at each time step
        for t in range(T):
            x_t = x[:, t, :]           # index into time-step
            h_t = self.cell(x_t, h_t)  # recur through GRU
            output_t = self.proj(h_t)  # project to vocab_size

            seq_outputs.append(output_t)
                    
        logits = torch.stack(seq_outputs, dim=1)  # B, T, vs

        targets = target_seq.reshape(-1).to(dtype=torch.long)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets, ignore_index=0)

        return logits, loss

    def predict(self, phonetic_input, max_len: int = 18):
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
                logits_t = self.proj(h_t)  # B, vocab_size := 32

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
    
