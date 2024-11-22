import pandas as pd
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer



class regressoionHead(nn.Module):

    def __init__(self, d_embedding: int):
        super().__init__()
        self.layer1 = nn.Linear(d_embedding, d_embedding//2)
        self.layer2 = nn.Linear(d_embedding//2, d_embedding//4)
        self.layer3 = nn.Linear(d_embedding//4, d_embedding//8)
        self.layer4 = nn.Linear(d_embedding//8, 1)
        self.relu=nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        
        return self.layer4(x)


class Transformer(nn.Module):

    def __init__(self,  d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'
 
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = Encoder(d_model)
        self.d_model = d_model


    def forward(self, src: Tensor, frac) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.token_encoder(src, frac)
        print(src)
        output = self.transformer_encoder(src)
        return output

class TransformerRegressor(nn.Module):

    def __init__(self, transformer, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.transformer = transformer
        self.regressionHead = regressoionHead(d_model)

    def init_weights(self) -> None:
        # initrange = 0.1
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        nn.init.xavier_normal_(self.regressionHead.weight)

    def forward(self, src: Tensor, frac) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        output = self.transformer(src, frac)
        output = self.regressionHead(output[:, 0:1, :])
        return output

def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class TransformerPretrain(nn.Module):

    def __init__(self,  d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.1):
        super().__init__()
        self.model_type = 'Transformer'

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = Encoder(d_model)
        self.d_model = d_model


    def forward(self, src: Tensor, frac) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.token_encoder(src, frac) 

        output = self.transformer_encoder(src)
        output_embed = output[:, 0:1, :]
        output_embed_proj = output_embed.squeeze(1)
        return output_embed_proj
    
    
class Embedder(nn.Module):
    def __init__(self,
                 d_model):
        super().__init__()
        self.d_model = d_model
        

        #elem_dir = '/MCIRLM'
        # # Choose what element information the model receives
        mat2vec = './mat2vec.csv'  # element embedding

        
        cbfv = pd.read_csv(mat2vec, index_col=0).values
        feat_size = cbfv.shape[-1]
        self.fc_mat2vec = nn.Linear(feat_size, d_model)
        zeros = np.zeros((1, feat_size))
        cat_array = np.concatenate([zeros, cbfv])
        cat_array = torch.as_tensor(cat_array, dtype=torch.float32)
        self.cbfv = nn.Embedding.from_pretrained(cat_array) \
            .to(dtype=torch.float32)

    def forward(self, src):
        mat2vec_emb = self.cbfv(src)
        x_emb = self.fc_mat2vec(mat2vec_emb)
        return x_emb


class Encoder(nn.Module):
    def __init__(self,
                 d_model,
                 frac=False,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model
        self.fractional = frac
        self.embed = Embedder(d_model=self.d_model)
        self.pe = FractionalEncoder(self.d_model, resolution=5000, log10=False)
        self.ple = FractionalEncoder(self.d_model, resolution=5000, log10=True)

        self.emb_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler = nn.parameter.Parameter(torch.tensor([1.]))
        self.pos_scaler_log = nn.parameter.Parameter(torch.tensor([1.]))



    def forward(self, src, frac):
        x = self.embed(src) * 2**self.emb_scaler
        pe = torch.zeros_like(x)
        ple = torch.zeros_like(x)
        pe_scaler = 2**(1-self.pos_scaler)**2
        ple_scaler = 2**(1-self.pos_scaler_log)**2
        pe[:, :, :self.d_model//2] = self.pe(frac) * pe_scaler
        ple[:, :, self.d_model//2:] = self.ple(frac) * ple_scaler
        x = x + pe + ple



        return x

class FractionalEncoder(nn.Module):
    """
    Encoding element fractional amount using a "fractional encoding" inspired
    by the positional encoder discussed by Vaswani.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self,
                 d_model,
                 resolution=100,
                 log10=False,
                 compute_device=None):
        super().__init__()
        self.d_model = d_model//2
        self.resolution = resolution
        self.log10 = log10
        self.compute_device = compute_device

        x = torch.linspace(0, self.resolution - 1,
                           self.resolution,
                           requires_grad=False) \
            .view(self.resolution, 1)
        fraction = torch.linspace(0, self.d_model - 1,
                                  self.d_model,
                                  requires_grad=False) \
            .view(1, self.d_model).repeat(self.resolution, 1)

        pe = torch.zeros(self.resolution, self.d_model)
        pe[:, 0::2] = torch.sin(x /torch.pow(
            50,2 * fraction[:, 0::2] / self.d_model))
        pe[:, 1::2] = torch.cos(x / torch.pow(
            50, 2 * fraction[:, 1::2] / self.d_model))
        pe = self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.clone()
        if self.log10:
            x = 0.0025 * (torch.log2(x))**2
            x = torch.clamp(x, max=1)

        x = torch.clamp(x, min=1/self.resolution)
        frac_idx = torch.round(x * (self.resolution)).to(dtype=torch.long) - 1
        out = self.pe[frac_idx]

        return out