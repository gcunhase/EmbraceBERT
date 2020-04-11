import torch
from torch import nn
from torch.nn import Parameter, init
from torch.autograd import Variable
import numpy as np
from torchnlp.nn.attention import Attention


class SelfAttention(nn.Module):
    """Code heavily based on https://gist.github.com/cbaziotis/94e53bdd6e4852756e0395560ff38aa4"""
    def __init__(self, attention_size, batch_first=False, non_linearity="tanh"):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = Parameter(torch.FloatTensor(attention_size))
        self.softmax = nn.Softmax(dim=-1)

        if non_linearity == "relu":
            self.non_linearity = nn.ReLU()
        else:
            self.non_linearity = nn.Tanh()

        init.uniform(self.attention_weights.data, -0.005, 0.005)

    def forward(self, inputs, attention_mask=None):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        # inputs = np.array(inputs)  #.detach().numpy()  # (bs, 128, 768)
        mat_mul = np.matmul(inputs, self.attention_weights.cpu().detach().numpy())  # att_weights = [768]
        mat_mul = torch.tensor(mat_mul, dtype=torch.float)
        scores = self.non_linearity(mat_mul)  # (bs, 128)
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        # attention_mask is already in the correct format
        if attention_mask is not None:
            mask = torch.tensor(attention_mask, dtype=torch.float)

            # apply the mask - zero out masked timesteps
            masked_scores = scores * mask  # (bs, 128)

            # re-normalize the masked scores
            _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
            scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        inputs_tensor = torch.tensor(inputs, dtype=torch.float)
        weighted = torch.mul(inputs_tensor, scores.unsqueeze(-1).expand_as(inputs_tensor))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores


class SelfAttentionPytorch(nn.Module):
    """ Compare each word in query with context

        Tokens to embrace = Context = [bs, 128, 768]
        Query=[bs, 1, 768]
    """
    def __init__(self, hidden_size):
        super(SelfAttentionPytorch, self).__init__()
        self.hidden_size = hidden_size
        self.attention_layer = Attention(self.hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tokens_to_embrace_bs):  #query, context):
        context = tokens_to_embrace_bs.permute(0, 2, 1).contiguous()
        context_size = context.size()
        selfattention_scores = torch.zeros([context_size[0], 1, context_size[2]])
        for i in range(0, context_size[1]):
            query = context[:, i, :]
            query = query.unsqueeze(1)
            attention_score_word, _ = self.attention_layer(query, context)
            selfattention_scores = torch.add(selfattention_scores, attention_score_word.squeeze().cpu())

        selfattention_scores = selfattention_scores / selfattention_scores.max()

        selfattention_scores = self.softmax(selfattention_scores)
        selfattention_scores = selfattention_scores.squeeze().squeeze()
        return selfattention_scores
