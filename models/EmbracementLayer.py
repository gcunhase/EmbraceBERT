import torch
from torch import nn
import numpy as np
from torchnlp.nn.attention import Attention
# torchnlp.Attention: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html

__author__ = "Gwena Cunha"


class EmbracementLayer(nn.Module):
    def __init__(self, hidden_size):
        super(EmbracementLayer, self).__init__()
        self.hidden_size = hidden_size
        self.embrace_attention = Attention(self.hidden_size)

    def forward(self, bert_output):
        # pooled_enc_output = bs x 768
        output_tokens_from_bert = bert_output[0]
        cls_output = bert_output[1]  # CLS

        # Docking layer not needed given that all features have the same size
        # Embracement layer with all outputs (except CLS)
        tokens_to_embrace = output_tokens_from_bert[:, 1:, :]  # (8, 128, 768) = (bs, sequence_length (where the first index is CLS), embedding_size)
        [bs, seq_len, emb_size] = tokens_to_embrace.size()
        tokens_to_embrace = tokens_to_embrace.cpu().detach().numpy()
        # Consider each token in the sequence of 128 as one modality.
        # 1. Multinomial distribution: randomly chose features from all 128 with same probability for each index feature
        probability = torch.tensor(np.ones(seq_len), dtype=torch.float)
        embraced_features_index = torch.multinomial(probability, emb_size, replacement=True)
        embraced_features_index = embraced_features_index.cpu().detach().numpy()
        # 2. Add features into one of size (bs, embedding_size)
        embraced_features_token = []
        for i_bs in range(bs):
            embraced_features_token_bs = []
            for i_emb, e in enumerate(embraced_features_index):
                embraced_features_token_bs.append(tokens_to_embrace[i_bs, e, i_emb])
            embraced_features_token.append(embraced_features_token_bs)
        embraced_features_token = torch.tensor(embraced_features_token, dtype=torch.float)

        # 3. Apply attention layer to CLS and embraced_features_token
        query = torch.unsqueeze(cls_output, 1).cuda()  # query = torch.randn(5, 1, 256)
        context = torch.unsqueeze(embraced_features_token, 1).cuda()  # context = torch.randn(5, 5, 256)
        embrace_output, weights = self.embrace_attention(query, context)
        embrace_output = embrace_output.squeeze()

        return embrace_output