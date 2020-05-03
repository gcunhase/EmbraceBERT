import torch
from torch import nn
from torchnlp.nn.attention import Attention
# torchnlp.Attention: https://pytorchnlp.readthedocs.io/en/latest/_modules/torchnlp/nn/attention.html

__author__ = "Gwena Cunha"


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.embrace_attention = Attention(self.hidden_size)

    def forward(self, cls_output, embraced_features_token, unsqueeze_idx=1):
        # Last stage: Apply attention layer to CLS and embraced_features_token
        query = torch.unsqueeze(cls_output, unsqueeze_idx).cuda()  # query = torch.randn(5, 1, 256)
        context = torch.unsqueeze(embraced_features_token, unsqueeze_idx).cuda()  # context = torch.randn(5, 5, 256)
        embrace_output, weights = self.embrace_attention(query, context)
        embrace_output = embrace_output.squeeze()

        return embrace_output