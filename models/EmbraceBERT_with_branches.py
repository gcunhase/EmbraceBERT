import torch
import torch.nn as nn
# from pytorch_transformers import *
from pytorch_transformers import BertModel, modeling_bert

from torch.nn import CrossEntropyLoss, MSELoss
from models.EmbracementLayer import EmbracementLayer
from models.CondensedEmbracementLayer import CondensedEmbracementLayer
from models.bert_utils import BertPreTrainedModel
from models.AttentionLayer import AttentionLayer

import numpy as np


# class BertForSequenceClassification(BertPreTrainedModel):
class EmbraceBertWithBranchesForSequenceClassification(BertPreTrainedModel):
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
        >>> config = BertConfig.from_pretrained('bert-base-uncased')
        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>>
        >>> model = BertForSequenceClassification(config)
        >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        >>> labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        >>> outputs = model(input_ids, labels=labels)
        >>> loss, logits = outputs[:2]
    """
    def __init__(self, config, dropout_prob, is_condensed=False, share_branch_classifier_weights=True):
        super(EmbraceBertWithBranchesForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_labels_evaluator = 2
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size  # 768
        self.is_condensed = is_condensed
        self.share_branch_classifier_weights = share_branch_classifier_weights
        # self.args = args

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout_prob)  # config.hidden_dropout_prob)
        if not self.is_condensed:
            self.embracement_layer = EmbracementLayer()
        else:
            self.embracement_layer = CondensedEmbracementLayer()
        self.embrace_attention = AttentionLayer(self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

        """BEGIN MODIFICATION
           Vector with classifiers # CHECK HOW MANY
           From original code: "We 'pool' the model by simply taking the hidden state corresponding to the first token." 
        """
        self.num_encoder_layers = config.num_hidden_layers

        # Check is the branches classifiers have shared weight
        if self.share_branch_classifier_weights:
            self.pooler_branches = modeling_bert.BertPooler(config)
            self.classifier_branches = nn.Linear(self.hidden_size, self.num_labels)
        else:
            self.pooler_branches = []
            self.classifier_branches = []
            for layer in range(self.num_encoder_layers):
                self.pooler_branches.append(modeling_bert.BertPooler(config))
                self.classifier_branches.append(nn.Linear(self.hidden_size, self.num_labels))

        # Classifier performance evaluator is the same for every layer (same functionality as having shared weights)
        self.evaluator_classifier = nn.Linear(self.num_labels, self.num_labels_evaluator)  # classes: good/bad
        self.embracement_layer_cls_with_branches = EmbracementLayer()
        """END MODIFICATION"""

        # Freeze BERT's weights
        #if freeze_berts_weights:
        #    self.bert.requires_grad = not freeze_berts_weights

        self.apply(self.init_weights)
        # self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, apply_dropout=False, freeze_bert_weights=False):

        # Fine-tune with
        if freeze_bert_weights:
            # self.bert.requires_grad = not freeze_bert_weights
            self.bert.training = not freeze_bert_weights

        # encoder_output = bs x 128 x 768
        bert_output = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                attention_mask=attention_mask, head_mask=head_mask)

        # pooled_enc_output = bs x 768
        # output_tokens_from_bert = bert_output[0]
        cls_output = bert_output[1]  # CLS
        output_tokens_from_bert = bert_output[0]
        hidden_tokens_from_bert = bert_output[2]

        """BEGIN MODIFICATION
           1. Add branches in hidden layers (BERT_base has 12 encoder layers + embedding layer = 13 layers)
           2. Pool the first token in each hidden state
           3. Classify
           4. Fine-tune ideal H^T_m (exit entropy) for decision making during inference
        """
        logits_branches = []
        logits_branches_evaluator = []
        labels_branch_evaluator = []
        pooled_output_cls_with_branches = []  # cls_output.unsqueeze(1)
        for l in range(1, self.num_encoder_layers+1):
            # Get tokens from hidden_layer 'l'
            hidden_token = hidden_tokens_from_bert[l].detach().cpu()

            # Pool first token + fully connected layer (dense layer) + activation (tanh)
            if self.share_branch_classifier_weights:
                pooled_output = self.pooler_branches(hidden_token.cuda())
            else:
                pooled_output = self.pooler_branches[l-1](hidden_token)

            # Classify pooled hidden token
            # Check is the branches classifiers have shared weight
            if self.share_branch_classifier_weights:
                logits_branch = self.classifier_branches(pooled_output.cuda())
            else:
                logits_branch = self.classifier_branches[l - 1](pooled_output)
            logits_branches.append(logits_branch.cuda())

            # Prediction
            pred_branch = logits_branch.detach().cpu().numpy()
            pred_branch = np.argmax(pred_branch, axis=1)
            lab_branch = labels.detach().cpu().numpy()
            lab_branch_evaluator = (pred_branch == lab_branch)
            labels_branch_evaluator.append(torch.tensor(lab_branch_evaluator.astype('uint8'), dtype=torch.long).cuda())

            # Fine-tune evaluator during training (this is to substitute the entropy check, automatic instead of static).
            #  Allows for better prediction of unseen samples (according to BranchyNet paper).
            logits_branch_evaluator = self.evaluator_classifier(logits_branch.cuda())
            logits_branches_evaluator.append(logits_branch_evaluator)

            # This means that ALL samples in the batch should have been correctly classified to be given as token to
            #  the embracement layer. This allows for additional incompleteness/uncertainty
            if lab_branch_evaluator.all():
                new_pooled_output = pooled_output.unsqueeze(1)
                if len(pooled_output_cls_with_branches) == 0:  # len(pooled_output_cls_with_branches.size()) == 0:
                    pooled_output_cls_with_branches = new_pooled_output
                else:
                    pooled_output_cls_with_branches = torch.cat(
                        (pooled_output_cls_with_branches, new_pooled_output), 1)  # .detach().cuda()

        # attention_mask_branches = torch.tensor(attention_mask_branches, dtype=torch.int64)
        if len(pooled_output_cls_with_branches) != 0:
            [bs, seq_len, _] = pooled_output_cls_with_branches.size()
            attention_mask_branches = torch.ones([bs, seq_len], dtype=torch.int64)
            attention_mask = torch.cat((attention_mask_branches, attention_mask.detach().cpu()), 1).cuda()
            output_tokens_from_bert = torch.cat((output_tokens_from_bert, pooled_output_cls_with_branches), 1)
        # New EmbraceLayer with cls_token and pooled_output_branches
        # embraced_cls_with_branches = self.embracement_layer_cls_with_branches(pooled_output_cls_with_branches)
        """END MODIFICATION"""

        if self.is_condensed:  # Embracement layer with outputs between CLS and SEP only
            embraced_features_token = self.embracement_layer(output_tokens_from_bert, attention_mask)
        else:  # Embracement layer with all outputs (except CLS)
            embraced_features_token = self.embracement_layer(output_tokens_from_bert)

        # Last step: Apply attention layer to CLS and embraced_features_token
        # embrace_output = self.embrace_attention(embraced_cls_with_branches, embraced_features_token)
        embrace_output = self.embrace_attention(cls_output, embraced_features_token)

        # No need because the embrace layer function as a dropout mechanism?
        if apply_dropout:
            embrace_output = self.dropout(embrace_output)

        # Classify
        logits = self.classifier(embrace_output)

        # a_bert_output = bert_output[2:]
        # a_embrace_output = embrace_output
        outputs = (logits,) + bert_output[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                """BEGIN MODIFICATION
                   Add losses from all branches
                """
                loss_branches = 0
                r_l = 0
                r_u = 1
                for l in range(1, self.num_encoder_layers+1):
                    logits_branch = logits_branches[l-1]
                    alpha = r_l + (r_u - r_l)/l
                    loss_branches += alpha*loss_fct(logits_branch.view(-1, self.num_labels), labels.view(-1))

                loss_branches_evaluator = 0
                for l in range(1, self.num_encoder_layers+1):
                    logits_branch_evaluator = logits_branches_evaluator[l-1]
                    label_branch_evaluator = labels_branch_evaluator[l-1]
                    loss_branches_evaluator += loss_fct(logits_branch_evaluator.view(-1, self.num_labels_evaluator), label_branch_evaluator.view(-1))

                loss += loss_branches + loss_branches_evaluator
                """END MODIFICATION"""

            outputs = (loss,) + outputs  # loss and probability of each class (vector)

        return outputs  # (loss), logits, (hidden_states), (attentions)
