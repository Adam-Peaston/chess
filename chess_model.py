import numpy as np
import math, pickle, os
from itertools import chain
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset

from chess_primitives import points_balance

class MultiHeadAttention(nn.Module):
    """
    Implements MultiHeadAttention in one shot, without calling on sub-layers for each attention head. Maybe more efficient.
    Splits up input embed_dim accross each of the 
    """
    def __init__(self, nheads, embed_dim, dk=None, masked=False, device=None):
        super(MultiHeadAttention, self).__init__()
        self.nheads, self.embed_dim, self.dk, self.masked, self.device = nheads, embed_dim, dk, masked, device

        # Allow user-specified or automatically set.
        if self.dk is None:
            # Allowing embed_dim to be non-integer multiple of nheads requires linear output transformation.
            self.dk = math.ceil(embed_dim / nheads) # Not setting dk splits embed dim near equally accross the nheads

        # Only implement linear output layer if embed_dim != nheads * dk, otherwise unnecessary.
        self.Wo = None
        if embed_dim != nheads*self.dk:
            self.Wo = nn.Parameter(torch.empty(nheads*self.dk, embed_dim, device=self.device))

        # Multi-head Wq, Wk, Wv weight tensor
        self.MhWqkv = nn.Parameter(torch.empty(embed_dim, nheads*3*self.dk, device=self.device))
        self.softmax = nn.Softmax(dim=3)
        self.init_model()

    def init_model(self):
        nn.init.xavier_uniform_(self.MhWqkv, gain=0.1)
        if isinstance(self.Wo, torch.Tensor):
            nn.init.xavier_uniform_(self.Wo, gain=0.1)
        self.to(self.device)

    def forget(self, p=0.01, gain=0.1):
        Wo_mask = np.random.choice((True,False),np.product(self.Wo.shape),p=[p,(1-p)]).reshape(self.Wo.shape)
        self.Wo[Wo_mask] = torch.rand(Wo_mask.sum(), device=self.device) * gain

    def forward(self, x):
        """Arguments: Input array 'x' of shape (N, W, E)."""
        mask = torch.zeros((x.shape[1], x.shape[1]), device=self.device)
        if self.masked:
            mask[torch.triu_indices(x.shape[1],1)] = -float('inf')

        HQKV = x @ self.MhWqkv                                                          # (N, W, E) @ (E, nheads * 3 * dk) = (N, W, nheads * 3 * dk)
        HQKVhat = HQKV.reshape(HQKV.shape[0], HQKV.shape[1], self.nheads, 3*self.dk)    # (N, W, nheads * 3 * dk) => (N, W, nheads, 3 * dk)
        HQKVhat = HQKVhat.transpose(1,2)                                                # (N, W, nheads, 3 * dk) => (N, nheads, W, 3 * dk)
        HQ, HK, HV = torch.split(HQKVhat, self.dk, dim=3)                               # Separate into Q, K, V;  3 * (N, nheads, W, dk)
        HI = (HQ @ HK.transpose(2,3) + mask)/(self.dk**0.5)                             # (N, nheads, Wq, dk) @ (N, nheads, dk, Wk) = (N, nheads, Wq, Wk)
        HS = self.softmax(HI)                                                           # (N, nheads, Wq, Wk) - sum over each row of stacked matrices == 1
        HA = HS @ HV                                                                    # (N, nheads, Wq, Wk) @ (N, nheads, Wv, dk) => (N, nheads, Wq, dk)
        HA = HA.transpose(1,2)                                                          # (N, nheads, Wq, dk) => (N, Wq, nheads, dk)
        HA = HA.reshape(x.shape[0],x.shape[1],-1)                                       # (N, Wq, nheads, dk) => (N, Wq, nheads * dk)
        if isinstance(self.Wo, torch.Tensor):
            z = HA @ self.Wo                                                            # (N, Wq, nheads * dk) @ (nheads * dk, E) = (N, Wq, E)
        else:
            z = HA                                                                      # (N, Wq, nheads * dk) => (N, Wq, E)
        return z

class FeedForward(nn.Module):
    def __init__(self, embed_dim, device=None):
        super(FeedForward, self).__init__()
        self.embed_dim, self.device = embed_dim, device
        self.lin1 = nn.Linear(embed_dim, embed_dim, device=device)
        self.lrelu = nn.LeakyReLU()
        self.lin2 = nn.Linear(embed_dim, embed_dim, device=device)
        self.feedforward = nn.Sequential(self.lin1, self.lrelu, self.lin2)
        self.init_model()
        
    def init_model(self):
        nn.init.xavier_uniform_(self.lin1.weight, gain=0.1)
        nn.init.xavier_uniform_(self.lin2.weight, gain=0.1)
        nn.init.constant_(self.lin1.bias.data, 0)
        nn.init.constant_(self.lin2.bias.data, 0)
        self.to(self.device)
    
    def forward(self, inputs):
        return self.feedforward(inputs)

class TransformerBlock(nn.Module):
    def __init__(self, nheads, embed_dim, dk=None, masked=False, device=None):
        self.nheads, self.embed_dim, self.dk, self.masked, self.device = nheads, embed_dim, dk, masked, device
        super(TransformerBlock, self).__init__()
        self.mha = MultiHeadAttention(nheads, embed_dim, dk=dk, masked=masked, device=device)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.init_model()

    def init_model(self):
        self.to(self.device)
    
    def forward(self, x):
        z = self.mha(x) + x # Skip connection
        z = self.ln1(z)
        z = self.ff(z) + z # Skip connection
        z = self.ln2(z)
        return z

class ChessAI(nn.Module):
    def __init__(self, nlayers, nheads, embed_dim, load_path=None, dk=None, masked=False, device=None):
        super(ChessAI, self).__init__()
        self.nlayers, self.nheads, self.embed_dim, self.dk, self.masked, self.device = nlayers, nheads, embed_dim, dk, masked, device
        self.embedder = nn.Embedding(22, embed_dim, device=device) # 22 possible states for each board square plus classifier token.
        self.tfm_layers = [TransformerBlock(nheads, embed_dim, dk=dk, masked=masked, device=device) for _ in range(nlayers)]
        self.model = nn.Sequential(*self.tfm_layers)
        self.scorer = nn.Linear(embed_dim, 1, bias=False, device=device)

        # Define positional encoding tensor statically
        position = torch.arange(64, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(64, 1, embed_dim, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0,1) # (S, 1, E) -> (1, S, E)
        self.register_buffer('pe', pe) # (1, S, E)

        # Conditionally initialize new model or load from passed path.
        if load_path is None:
            self.init_model()
        else:
            self.load_state_dict(torch.load(load_path))

    def init_model(self):
        initrange = 0.1
        self.embedder.weight.data.uniform_(-initrange, initrange)
        self.scorer.weight.data.uniform_(-initrange, initrange)
        self.to(self.device)

    def forward(self, x):
        '''Input is a tensor of shape (Board, Square) where the Square dimension is always 64 long and contains an integer representation of the square state.'''
        # Generate tokens which will accumulate the board scoring information, one per board.
        stokens = torch.ones((len(x),1), dtype=torch.int, device=self.device) * 21 # Assign scoring token to number 21 
        # Appending scoring tokens to the input tensor.
        x = torch.cat((x, stokens), dim=1)
        # Convert integer representations to embedding vectors.
        z = self.embedder(x) # (B, S) => (B, S, E)
        # Add positional encodings to all but the scoring token vectors
        z[:,:-1,:] += self.pe # (B, S, E) + (1, S, E)
        # Pass through model.
        z = self.model(z)
        # extract off the scoring token vector.
        z = z[:,-1,:].squeeze() # (B, E)
        # Simple linear scorer applies dot product and bias to each scoring token vector.
        z = self.scorer(z) # (B)
        # Return predictions of end scores from each board
        z = torch.tanh(z).flatten()
        return z # (B)
    
    def evaluate(self, x):
        with torch.no_grad():
            # convert x from numpy array to tensor on device
            x_tensor = torch.tensor(x, device=self.device)
            # forward pass without gradient
            z = self.forward(x_tensor).cpu().numpy()
            return z

class TransformerModel(nn.Module):
    def __init__(self, embed_dim, nheads, dk, nlayers, dropout=0.0, device=None, load_path=None):
        super(TransformerModel, self).__init__()
        self.device = device
        self.embedder = nn.Embedding(22, embed_dim, device=device)
        encoder_layer = TransformerEncoderLayer(embed_dim, nheads, dk, dropout, batch_first=True, device=device)
        self.transformer_encoder = TransformerEncoder(encoder_layer, nlayers)
        self.scorer = nn.Linear(embed_dim, 1, bias=False, device=device)

        # Define positional encoding tensor statically
        position = torch.arange(64, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, device=device) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(64, 1, embed_dim, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0,1) # (S, 1, E) -> (1, S, E)
        self.register_buffer('pe', pe) # (1, S, E)

        # Conditionally initialize new model or load from passed path.
        if load_path is None:
            self.init_weights()
        else:
            self.load_state_dict(torch.load(load_path))

    def init_weights(self):
        initrange = 0.1
        self.embedder.weight.data.uniform_(-initrange, initrange)
        self.scorer.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x):
        '''Input is a tensor of shape (Board, Square) where the Square dimension is always 64 long and contains an integer representation of the square state.'''
        # Generate tokens which will accumulate the board scoring information, one per board.
        scoring_tokens = torch.ones((len(x),1), dtype=torch.int, device=self.device) * 21 # Assign scoring token to number 21 
        # Appending scoring tokens to the input tensor.
        x = torch.cat((x, scoring_tokens), dim=1) # (B, S)
        # Convert integer representations to embedding vectors.
        z = self.embedder(x) # (B, S) => (B, S, E)
        # Add positional encodings to all but the scoring token vectors
        z[:,:-1,:] += self.pe # (B, S, E) + (1, S, E) => (B, S, E)
        # Add zero vector to each board for "ghostmax" ? (B, 1, E), essentially an additional square with zero embedding vector.
        ghost = torch.zeros((z.shape[0], 1, z.shape[2]), device=self.device)
        z = torch.cat((z, ghost), axis=1)
        # Forward through model.
        z = self.transformer_encoder(z)
        # extract off the scoring token vector.
        z = z[:,-1,:].squeeze() # (B, E)
        # Simple linear scorer applies dot product to each scoring token vector.
        z = self.scorer(z) # (B)
        # Return predictions of end scores from each board [-1, 1]
        z = torch.tanh(z).flatten()
        return z # (B)
    
    def evaluate(self, x):
        # forward pass without gradient
        with torch.no_grad():
            # convert x from numpy array to tensor on device
            x_tensor = torch.tensor(x, device=self.device, requires_grad=False)
            z = self.forward(x_tensor)
            # move back to cpu as numpy array
            z_numpy = z.cpu().numpy()
        return z_numpy
    
class PiecesModel(nn.Module):
    # This model type uses the points_balance function to evaluate positions - essentially based on material
    def __init__(self, device=None):
        super(PiecesModel, self).__init__()
        self.device = device
        
    def forward(self, x):
        scores, _ = zip(*[points_balance(b) for b in x])
        return torch.tensor(np.stack(scores), device=self.device)
        
    def evaluate(self, x):
        scores, _ = zip(*[points_balance(b) for b in x])
        return np.stack(scores)

## Helper function for the ChessDataset class

def compile_dataset(root_dir, look_back=10):
    # Catalogue training rounds to source dataset from
    training_round_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
    lookback_rounds = training_round_dirs[-look_back:]

    # compile positions
    # data[token] = {'board': board, 'visits': 1, 'points': points}
    data = {}
    for training_round_dir in lookback_rounds:
        self_play_path = os.path.join(root_dir, training_round_dir, 'self_play')
        tournamentfiles = [f for f in os.listdir(self_play_path) if f.startswith('tmnt_') and f.endswith('.pkl')]
        for file in tournamentfiles:
            with open(os.path.join(self_play_path, file), 'rb') as pkl:
                tourn = pickle.load(pkl)
            for i, pair in tourn.items():
                for order,game in pair.items():
                    for color in game:
                        points = game[color]['points']
                        for token,board in game[color]['moves']:
                            if token in data:
                                data[token]['visits'] += 1
                                data[token]['points'] += points
                            else:
                                data[token] = {'board': board, 'visits':1, 'points':points}
        
        # augment with checkmates from all previous training rounds
        for training_round_dir in training_round_dirs:
            checkmates_file = os.path.join(root_dir, training_round_dir, 'checkmates.pkl')
            with open(checkmates_file, 'rb') as pkl:
                checkmates = pickle.load(pkl)
            for token,board,points in checkmates:
                if token in data:
                    data[token]['visits'] += 1
                    data[token]['points'] += points
                else:
                    data[token] = {'board':board, 'visits':1, 'points':points}
    return data

class ChessDataset(Dataset):
    def __init__(self, root_dir, look_back, device):
        # Compile dataset from passed self_play directories
        self.data = compile_dataset(root_dir, look_back)
        self.catalogue = list(self.data.keys())
        self.device = device

    def __len__(self):
        return len(self.catalogue)

    def __getitem__(self, idx):
        # Obtain the dict containing the board, visits, and points objects 
        example = self.data[self.catalogue[idx]]
        X = torch.tensor(example['board'].flatten(), device=self.device)
        y = torch.tensor(example['points'] / example['visits'], device=self.device)
        return X, y


class TanhLoss(nn.Module):
    '''This class wraps nn.CrossEntropyLoss, re-scaling z and y to pretend they're actually probabilities'''
    def __init__(self, weighted=False):
        super(TanhLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.weighted = weighted

    def forward(self, args):
        if self.weighted is True:
            z, y, w = args
            return (self.criterion((z+1)/2, (y+1)/2) * w).sum()
        else:
            z, y = args
            return self.criterion((z+1)/2, (y+1)/2).sum()


def train(model, loss_fn, optimizer, train_dataloader, test_dataloader, warmup_passes, max_lr, save_dir, stopping):

    warm_steps = len(train_dataloader) * warmup_passes
    train_step = 0
    best_test_loss = float('inf')
    stopping_count = 0
    data_pass = 0

    while stopping_count < stopping:
        
        data_pass += 1
        loss_sum = 0
        examples_seen = 0

        for data in train_dataloader:
            optimizer.zero_grad()
            X, y = data
            z = model(X)
            loss = loss_fn((z, y))
            loss_sum += loss.item()
            examples_seen += X.shape[0]
            loss.backward()
            optimizer.step()
            # learning rate warm-up
            train_step += 1
            for g in optimizer.param_groups:
                g['lr'] = min(max_lr, max_lr * train_step / warm_steps)
        mean_train_loss = loss_sum / examples_seen

        with torch.no_grad():
            loss_sum = 0
            examples_seen = 0
            for data in test_dataloader:
                X, y = data
                z = model(X)
                loss = loss_fn((z, y))
                loss_sum += loss.item()
                examples_seen += X.shape[0]
            mean_test_loss = loss_sum / examples_seen

            if mean_test_loss < best_test_loss:
                torch.save(model.state_dict(), save_dir)
                best_test_loss = mean_test_loss
                stopping_count = 0
            else:
                stopping_count += 1

        print(f'Pass: {data_pass}, train loss: {mean_train_loss:,.5f}, test loss: {mean_test_loss:,.5f}, stopping count: {stopping_count}')
    return model