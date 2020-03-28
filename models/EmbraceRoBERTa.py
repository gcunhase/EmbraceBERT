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
"""PyTorch RoBERTa model. """

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
from models.EmbracementLayer import EmbracementLayer
from models.CondensedEmbracementLayer import CondensedEmbracementLayer
from models.AttentionLayer import AttentionLayer
from models.BranchesLayer import BranchesLayer

from models.roberta_utils import (ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,
                                  ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING,
                                  RobertaConfig, RobertaModel, RobertaClassificationHead)


logger = logging.getLogger(__name__)


@add_start_docstrings("""RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer 
    on top of the pooled output) e.g. for GLUE tasks. """,
                      ROBERTA_START_DOCSTRING, ROBERTA_INPUTS_DOCSTRING)
class EmbraceRobertaForSequenceClassification(BertPreTrainedModel):
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

    def __init__(self, config, dropout_prob, is_condensed=False, add_branches=False, share_branch_weights=False):
        super(EmbraceRobertaForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size  # 768
        self.is_condensed = is_condensed

        """EmbraceBERT with branches"""
        self.num_labels_evaluator = 2
        self.add_branches = add_branches
        self.share_branch_weights = share_branch_weights
        """END MODIFICATION"""

        self.roberta = RobertaModel(config)
        if not self.is_condensed:
            self.embracement_layer = EmbracementLayer()
        else:
            self.embracement_layer = CondensedEmbracementLayer()

        self.embrace_attention = AttentionLayer(self.hidden_size)
        self.classifier = RobertaClassificationHead(config, dropout_prob)
        # self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        """EmbraceBERT with branches"""
        if self.add_branches:
            self.branches_layer = BranchesLayer(config, share_branch_weights,
                                                num_labels_evaluator=self.num_labels_evaluator)
        """END MODIFICATION"""

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, apply_dropout=False, freeze_bert_weights=False):

        # Fine-tune with
        if freeze_bert_weights:
            self.roberta.requires_grad = not freeze_bert_weights

        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, head_mask=head_mask)
        cls_output = outputs[1]  # CLS
        output_tokens_from_bert = outputs[0]

        """EmbraceBERT with branches"""
        if self.add_branches:
            hidden_tokens_from_bert = outputs[2]
            output_tokens_from_bert, attention_mask, logits_branches, logits_branches_evaluator, labels_branch_evaluator = self.branches_layer(
                hidden_tokens_from_bert, output_tokens_from_bert, attention_mask, labels)
        """END MODIFICATION"""

        # Embracement layer with attention and no docking
        # sequence_output = outputs[0]
        if self.is_condensed:  # Embracement layer with outputs between CLS and SEP only
            embraced_features_token = self.embracement_layer(output_tokens_from_bert, attention_mask)
        else:  # Embracement layer with all outputs (except CLS)
            embraced_features_token = self.embracement_layer(output_tokens_from_bert)

        # Last step: Apply attention layer to CLS and embraced_features_token
        embrace_output = self.embrace_attention(cls_output, embraced_features_token)

        if len(embrace_output.shape) == 1:  # Last batch might have only 1 sample
            embrace_output = embrace_output.unsqueeze(0)
        embrace_output = embrace_output.unsqueeze(1)

        # Classify
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

            """EmbraceBERT with branches: Add losses from all branches"""
            if self.add_branches:
                loss_branches, loss_branches_evaluator = \
                    self.branches_layer.loss_branches_and_evaluator(loss_fct, logits_branches,
                                                                    logits_branches_evaluator, labels,
                                                                    labels_branch_evaluator)
                loss += loss_branches + loss_branches_evaluator
            """END MODIFICATION"""

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

