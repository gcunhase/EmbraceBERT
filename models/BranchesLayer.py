import torch
from torch import nn
from pytorch_transformers import modeling_bert
from models.EmbracementLayer import EmbracementLayer
import numpy as np

__author__ = "Gwena Cunha"


class BranchesLayer(nn.Module):
    def __init__(self, config, share_branch_weights, num_labels_evaluator=2):
        super(BranchesLayer, self).__init__()
        self.num_labels = config.num_labels
        self.num_labels_evaluator = num_labels_evaluator
        self.hidden_size = config.hidden_size
        self.share_branch_weights = share_branch_weights

        """ Vector with classifiers
            From original BERT code: "We 'pool' the model by simply taking the hidden state corresponding to the first token." 
        """
        self.num_encoder_layers = config.num_hidden_layers

        # Check is the branches classifiers have shared weight
        if self.share_branch_weights:
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
        # self.embracement_layer_cls_with_branches = EmbracementLayer()

    def loss_branches_and_evaluator(self, loss_fct, logits_branches, logits_branches_evaluator,
                                    labels, labels_branch_evaluator):
        loss_branches = 0
        r_l = 0
        r_u = 1
        for l in range(1, self.num_encoder_layers + 1):
            logits_branch = logits_branches[l - 1]
            alpha = r_l + (r_u - r_l) / l
            loss_branches += alpha * loss_fct(logits_branch.view(-1, self.num_labels), labels.view(-1))

        loss_branches_evaluator = 0
        for l in range(1, self.num_encoder_layers + 1):
            logits_branch_evaluator = logits_branches_evaluator[l - 1]
            label_branch_evaluator = labels_branch_evaluator[l - 1]
            loss_branches_evaluator += loss_fct(logits_branch_evaluator.view(-1, self.num_labels_evaluator),
                                                label_branch_evaluator.view(-1))

        return loss_branches, loss_branches_evaluator

    def forward(self, hidden_tokens_from_bert, output_tokens_from_bert, attention_mask, labels):
        """Description
           1. Add branches in hidden layers (BERT_base has 12 encoder layers + embedding layer = 13 layers)
           2. Pool the first token in each hidden state
           3. Classify
           4. Fine-tune ideal H^T_m (exit entropy) for decision making during inference
        """
        logits_branches = []
        logits_branches_evaluator = []
        labels_branch_evaluator = []
        pooled_output_cls_with_branches = []  # cls_output.unsqueeze(1)
        for l in range(1, self.num_encoder_layers + 1):
            # Get tokens from hidden_layer 'l'
            hidden_token = hidden_tokens_from_bert[l].detach().cpu()

            # Pool first token + fully connected layer (dense layer) + activation (tanh)
            if self.share_branch_weights:
                pooled_output = self.pooler_branches(hidden_token.cuda())
            else:
                pooled_output = self.pooler_branches[l - 1](hidden_token)

            # Classify pooled hidden token
            # Check is the branches classifiers have shared weight
            if self.share_branch_weights:
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

        return output_tokens_from_bert, attention_mask, logits_branches, logits_branches_evaluator, labels_branch_evaluator