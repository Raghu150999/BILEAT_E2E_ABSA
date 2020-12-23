import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from seq_utils import *
from bert import BertPreTrainedModel


class TaggerConfig:
    def __init__(self):
        self.hidden_dropout_prob = 0.1
        self.hidden_size = 768
        self.bidirectional = True  # not used if tagger is non-RNN model

class BERTLINEAR(nn.Module):

    def __init__(self, config):
        super(BERTLINEAR, self).__init__()
        self.config = config
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, x, attention_mask=None, labels=None, labels_normal=None, lm_labels=None, labels_sent=None, labels_op=None):
        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        logits = self.fc(x)
        active_loss = attention_mask.view(-1) == 1
        active_logits = logits.view(-1, self.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = nn.CrossEntropyLoss()(active_logits, active_labels)
        return loss, logits

class BILEAT(nn.Module):

    def __init__(self, config):
        super(BILEAT, self).__init__()
        self.config = config
        self.epsilon = 0.5
        self.d = 384
        self.loss_weight = 0.1
        self.fc_ae_1 = nn.Linear(config.hidden_size, self.d * 2)
        self.fc_op_1 = nn.Linear(config.hidden_size, self.d * 2)
        self.fc_ae = nn.Linear(self.d * 2, config.num_normal_labels)
        self.fc_op = nn.Linear(self.d * 2, config.num_normal_labels)
        self.fc_direct = nn.Linear(self.d * 2, config.num_labels)
        self.fc = nn.Linear(self.d * 8, config.num_labels)
        
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.1)
        
        self.W = torch.empty(self.d * 2, self.d * 2)
        nn.init.xavier_uniform_(self.W)
        self.W = self.W.to(config.device)

    def forward(self, x, attention_mask=None, labels=None, labels_normal=None, lm_labels=None, labels_sent=None, labels_op=None):

        h_ae = self.fc_ae_1(x) # Eq 4
        h_op = self.fc_op_1(x) # Eq 5

        # AE and OE auxiliary tasks
        o_ae = self.fc_ae(F.relu(h_ae)) 
        o_op = self.fc_op(F.relu(h_op)) 

        p_ae = self.softmax(o_ae) # Eq 6
        p_op = self.softmax(o_op) # Eq 7

        # B: 1, O: 2, Find probability of a word being part of an aspect term
        p_ae = p_ae[:, :, 1] + p_ae[:, :, 2] # (bsz, seq_len)
        p_ae = p_ae.unsqueeze(1) # (bsz, 1, seq_len)

        # Find probability of a word being part of an opinion term
        p_op = p_op[:, :, 1] + p_op[:, :, 2] # (bsz, seq_len)
        p_op = p_op.unsqueeze(1) # (bsz, 1, seq_len)

        seq_len = x.size()[1] # N
        zero_diag = -1e18 * torch.eye(seq_len, seq_len, requires_grad=False).to(self.config.device)
        
        idxs = torch.arange(0, seq_len, requires_grad=False).to(self.config.device)
        idxs = idxs.unsqueeze(1) # (seq_len, 1)
        tmp = idxs * torch.ones(seq_len, seq_len, requires_grad=False).to(self.config.device) # (seq_len, seq_len)
        dist_metric = torch.abs(tmp - tmp.transpose(0, 1)) + torch.eye(seq_len, seq_len, requires_grad=False).to(self.config.device) # (seq_len, seq_len)
        dist_metric = 1 / dist_metric

        A = h_ae @ self.W @ h_op.transpose(1, 2) # bsz, seq_len, seq_len
        A = A + zero_diag # (bsz, seq_len, seq_len)
        # Score matrix Q, Eq 8
        A = A * dist_metric
        
        op_prime = self.softmax(A * p_op) @ h_op # Eq 9 + 11
        ae_prime = self.softmax(A.transpose(1, 2) * p_ae) @ h_ae # Eq 10 + 12

        c = torch.cat([h_ae, ae_prime, h_op, op_prime], dim=2) # (bsz, seq_len, 4 * h), Eq 13
        o_prime = self.fc(c) # Eq 14

        # Loss computations
        loss = 0
        active_loss = attention_mask.view(-1) == 1

        # Aspect tag predictions (AE)
        active_logits = o_ae.view(-1, self.config.num_normal_labels)[active_loss]
        active_labels = labels_normal.view(-1)[active_loss]
        loss += self.loss_weight * nn.MultiMarginLoss(margin=1)(active_logits, active_labels)

        # Opinion tag predictions (OE)
        active_logits = o_op.view(-1, self.config.num_normal_labels)[active_loss]
        active_labels = labels_op.view(-1)[active_loss]
        loss += self.loss_weight * nn.MultiMarginLoss(margin=1)(active_logits, active_labels)

        # Unified tag predictions (U)
        active_logits = o_prime.view(-1, self.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss += nn.MultiMarginLoss(margin=3)(active_logits, active_labels)

        return loss, o_prime

class BertABSATagger(BertPreTrainedModel):
    def __init__(self, bert_config):
        """

        :param bert_config: configuration for bert model
        """
        super(BertABSATagger, self).__init__(bert_config)
        self.tagger_config = TaggerConfig()
        self.tagger_config.absa_type = bert_config.absa_type.lower()
        if bert_config.tfm_mode == 'finetune':
            self.bert = AutoModel.from_pretrained(bert_config.model_name_or_path, config=bert_config, cache_dir="./cache")
        else:
            raise Exception("Invalid transformer mode %s!!!" % bert_config.tfm_mode)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        # fix the parameters in BERT and regard it as feature extractor
        if bert_config.fix_tfm:
            # fix the parameters of the (pre-trained or randomly initialized) transformers during fine-tuning
            for p in self.bert.parameters():
                p.requires_grad = False
        self.tagger = BILEAT(bert_config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, labels_normal=None, lm_labels=None,
                labels_sent=None, labels_op=None, idxs=None):
        outputs = self.bert(
            input_ids, 
            position_ids=position_ids, 
            token_type_ids=token_type_ids,
            attention_mask=attention_mask, 
            head_mask=head_mask)
        embeddings = outputs[2][11] # using output of 11th transformer layer (last-second)
        x = self.bert_dropout(embeddings)
        loss, logits = self.tagger(x, attention_mask, labels, labels_normal, lm_labels, labels_sent, labels_op)
        outputs = (loss, logits,) + outputs[2:]
        return outputs
