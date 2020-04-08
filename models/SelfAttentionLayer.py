import torch
from torch import nn
from torch.nn import Parameter, init
from torch.autograd import Variable
import numpy as np

"""Code heavily based on https://gist.github.com/cbaziotis/94e53bdd6e4852756e0395560ff38aa4"""


class SelfAttention(nn.Module):
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
