import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MaskedMeanPooler(nn.Module):
    """ Computes mean over elements whose mask is 1"""
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, data, masks):
        # data... [B, max_seq_len, emb_size]
        # masks... [B, max_seq_len]
        masked_data = data * masks.unsqueeze(2)
        return torch.sum(masked_data, dim=self.dim)


class WeightedSumPooler(nn.Module):
    """ Computes a weighted combination of embeddings (including PAD!) by compressing embeddings into single numbers
        and renormalizing them. """
    def __init__(self, embedding_size, dim=1):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(embedding_size, out_features=1).to(DEVICE)

    def forward(self, data, masks):
        # data... [B, max_seq_len, emb_size]
        # masks... [B, max_seq_len]
        weights = F.softmax(self.linear(data), dim=self.dim)
        weighted_comb = torch.sum(weights * data, dim=self.dim)
        return weighted_comb  # [B, emb_size]


class LSTMPooler(nn.Module):
    """Applies LSTM over sequences where the mask is 1"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True).to(DEVICE)

    def forward(self, data, masks):
        # data... [B, max_seq_len, emb_size]
        # masks... [B, max_seq_len]
        batch_size, max_seq_len = masks.shape
        bool_masks = masks.bool()  # TODO: mask the PAD tokens somehow? (has problems with a CUDA runtime error)

        _, (last_hidden, _) = self.lstm(data)
        return last_hidden[0]  # [B, emb_size]
