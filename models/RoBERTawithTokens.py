# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from pytorch_transformers.modeling_bert import (BertConfig, BertEmbeddings,
                                                BertLayerNorm, BertModel,
                                                BertPreTrainedModel, gelu)

from pytorch_transformers.modeling_utils import add_start_docstrings

from models.roberta_utils import (ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
                                  RobertaConfig, RobertaModel, RobertaClassificationHead)
from models.AttentionLayer import AttentionLayer


"""
    Vanilla RoBERTa with option to apply dropout or not
    
    My modification:
        1. Apply dropout: boolean option
        2. Dropout probability in run_classifier.py argparse 
"""


logger = logging.getLogger(__name__)


#@add_start_docstrings("""RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer
#    on top of the pooled output) e.g. for GLUE tasks. """,
#                      ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class RobertaWithTokensForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = RoertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config, dropout_prob, is_condensed=False, add_branches=False,
                 share_branch_weights=False, max_seq_length=128, token_layer_type='robertawithatt',
                 do_calculate_num_params=False):
        super(RobertaWithTokensForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size  # 768
        self.is_condensed = is_condensed
        self.max_seq_length = max_seq_length  # 128
        self.token_layer_type = token_layer_type
        self.do_calculate_num_params = do_calculate_num_params

        self.roberta = RobertaModel(config)
        if self.token_layer_type == 'robertawithatt':  # Attention Layer
            self.embrace_attention = AttentionLayer(self.hidden_size)
        elif self.token_layer_type == 'robertawithprojectionatt':
            # Projection Layer
            self.projection_layer = nn.Linear(self.max_seq_length, 1)
            # Attention Layer with CLS token and token from projection layer
            self.embrace_attention = AttentionLayer(self.hidden_size)
        elif self.token_layer_type == 'robertawithattclsprojection':
            # Attention Layer with CLS token and 128 tokens
            self.embrace_attention = AttentionLayer(self.hidden_size)
            # Projection Layer
            self.projection_layer = nn.Linear(2, 1)
        else:  # Projection Layer
            self.projection_layer = nn.Linear(self.max_seq_length + 1, 1)
        self.classifier = RobertaClassificationHead(config, dropout_prob)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, apply_dropout=False, freeze_bert_weights=False):

        # Fine-tune with
        if freeze_bert_weights:
            self.roberta.requires_grad = not freeze_bert_weights

        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, head_mask=head_mask)
        cls_output = outputs[1]  # CLS
        output_tokens_from_roberta = outputs[0]

        embraced_features_token = output_tokens_from_roberta

        # Apply attention layer to CLS and embraced_features_token
        # embrace_output = self.embrace_attention(embraced_cls_with_branches, embraced_features_token)
        if self.token_layer_type == 'robertawithatt':
            embrace_output = self.embrace_attention(cls_output, embraced_features_token)
            embrace_output = embrace_output[0]
        elif self.token_layer_type == 'robertawithprojectionatt':
            # Projection layer -> projection vector
            tokens = embraced_features_token.permute((0, 2, 1))
            projection_output = self.projection_layer(tokens).squeeze()
            if len(projection_output.shape) == 1:
                projection_output = projection_output.unsqueeze(0)
            # Attention layer (T_CLS, T_all)
            embrace_output = self.embrace_attention(cls_output, projection_output)
            embrace_output = embrace_output[0]
        elif self.token_layer_type == 'robertawithattclsprojection':
            # Attention Layer
            embrace_output = self.embrace_attention(cls_output, embraced_features_token)
            embrace_output = embrace_output[0]
            # Concatenate cls_output and embrace_output (bs, seq+1, hidden_dim) -> (8, 129, 768)
            if len(embrace_output.shape) == 1:
                embrace_output = embrace_output.unsqueeze(0).unsqueeze(0)
            elif len(embrace_output.shape) == 2:
                embrace_output = embrace_output.unsqueeze(1)
            tokens = torch.cat((cls_output.unsqueeze(1), embrace_output), 1)
            tokens = tokens.permute((0, 2, 1))
            # Projection layer to obtain 1 feature vector for classification
            embrace_output = self.projection_layer(tokens).squeeze()
        else:  # Projection layer
            # Concatenate cls_output and embraced_features_token (bs, seq+1, hidden_dim) -> (8, 129, 768)
            tokens = torch.cat((cls_output.unsqueeze(1), embraced_features_token), 1)
            tokens = tokens.permute((0, 2, 1))
            # Projection layer to obtain 1 feature vector for classification
            embrace_output = self.projection_layer(tokens).squeeze()

        # Classify, dropout also applied here, but maybe not needed because the embrace layer functions as
        #   a dropout mechanism?
        s = embrace_output.shape
        if len(s) == 1:  # bs=1
            embrace_output = embrace_output.unsqueeze(0).unsqueeze(0)
        else:  # bs != 1
            embrace_output = embrace_output.unsqueeze(1)
        # logits = self.classifier(embrace_output, apply_dropout)
        if self.do_calculate_num_params:
            # args.do_calculate_num_params:
            logits = self.classifier(embrace_output)
        else:  # Regular run:
            logits = self.classifier(embrace_output, apply_dropout)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
