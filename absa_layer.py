import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from seq_utils import *
from bert import BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from crf import CRF


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

class CollaborativeLM(nn.Module):

    def __init__(self, config):
        super(CollaborativeLM, self).__init__()
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
        
    
    def combine_with_alpha(self, o, o_prime):
        # compute confidence values
        alpha = self.epsilon * torch.sum(o_prime ** 2, dim=2)
        o_prime = o_prime.transpose(0, 1)
        o_prime = o_prime.transpose(0, 2) # (num_labels, bsz, seq)

        o = o.transpose(0, 1)
        o = o.transpose(0, 2) # (num_labels, bsz, seq)

        o_tilde = alpha * o_prime + (1 - alpha) * o
        o_tilde = o_tilde.transpose(0, 2)
        o_tilde = o_tilde.transpose(0, 1)
        return o_tilde

    def forward(self, x, attention_mask=None, labels=None, labels_normal=None, lm_labels=None, labels_sent=None, labels_op=None):

        h_ae = self.fc_ae_1(x) # 3
        h_op = self.fc_op_1(x) # 3

        o_ae = self.fc_ae(F.relu(h_ae))
        o_op = self.fc_op(F.relu(h_op))

        p_ae = self.softmax(o_ae)
        p_op = self.softmax(o_op)

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
        A = A * dist_metric
        op_prime = self.softmax(A * p_op) @ h_op
        ae_prime = self.softmax(A.transpose(1, 2) * p_ae) @ h_ae

        c = torch.cat([h_ae, ae_prime, h_op, op_prime], dim=2) # (bsz, seq_len, 4 * h)
        o_prime = self.fc(c)

        # Loss computations
        loss = 0
        active_loss = attention_mask.view(-1) == 1

        # Boundary tag predictions
        active_logits = o_ae.view(-1, self.config.num_normal_labels)[active_loss]
        active_labels = labels_normal.view(-1)[active_loss]
        loss += self.loss_weight * nn.MultiMarginLoss(margin=1)(active_logits, active_labels)

        # Opinion tag predictions
        active_logits = o_op.view(-1, self.config.num_normal_labels)[active_loss]
        active_labels = labels_op.view(-1)[active_loss]
        loss += self.loss_weight * nn.MultiMarginLoss(margin=1)(active_logits, active_labels)

        # Unified tag predictions
        active_logits = o_prime.view(-1, self.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss += nn.MultiMarginLoss(margin=3)(active_logits, active_labels)

        return loss, o_prime

class InteractiveLM(nn.Module):

    def __init__(self, config):
        super(InteractiveLM, self).__init__()
        self.config = config
        self.epsilon = 0.5
        self.d = 384
        self.fc_normal = nn.Linear(self.d * 2, config.num_normal_labels)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        self.fc_sent = nn.Linear(self.d * 2, config.num_sent_labels)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0.1)
        self.lstm_normal = nn.LSTM(config.hidden_size, self.d, batch_first=True, bidirectional=True)
        self.lstm_sent = nn.LSTM(config.hidden_size, self.d, batch_first=True, bidirectional=True)
        self.tanh = nn.Tanh()
        self.W = torch.empty(self.d * 2, self.d * 2)
        nn.init.xavier_uniform_(self.W)
        self.W = self.W.to(config.device)
    
    def forward(self, x, attention_mask=None, labels=None, labels_normal=None, lm_labels=None, labels_sent=None, labels_op=None):
        # x : (bsz, seq, hidden_size)
        h_normal = self.dropout(self.lstm_normal(x)[0])
        h_sent = self.dropout(self.lstm_sent(x)[0])
        o_normal = self.fc_normal(h_normal)
        o_sent = self.fc_sent(h_sent)

        # Loss computation
        loss = 0
        active_loss = attention_mask.view(-1) == 1

        # Boundary tag predictions
        active_logits = o_normal.view(-1, self.config.num_normal_labels)[active_loss]
        active_labels = labels_normal.view(-1)[active_loss]
        loss += nn.CrossEntropyLoss()(active_logits, active_labels)

        # Sentiment tag predictions
        active_logits = o_sent.view(-1, self.config.num_sent_labels)[active_loss]
        active_labels = labels_sent.view(-1)[active_loss]
        loss += nn.CrossEntropyLoss()(active_logits, active_labels)

        h_sent = h_sent.transpose(1, 2)
        A = self.tanh(h_normal @ self.W @ h_sent) # (bsz, seq_len, seq_len)
        x_prime = A @ x

        o = self.fc(x_prime)
        # Unified tag predictions
        active_logits = o.view(-1, self.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss += nn.CrossEntropyLoss()(active_logits, active_labels)

        return loss, o

class InteractiveBG(nn.Module):

    def __init__(self, config):
        super(InteractiveBG, self).__init__()
        self.config = config
        self.epsilon = 0.5
        self.fc_normal = nn.Linear(config.hidden_size, config.num_normal_labels)
        self.fc = nn.Linear(config.hidden_size, config.num_labels)
        self.fc_sent = nn.Linear(config.hidden_size, config.num_sent_labels)
        self.fc_mat = nn.Linear(config.num_normal_labels * config.num_sent_labels, config.num_labels)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0)
        # Interaction tensor
        self.G = torch.empty(config.num_labels, config.num_sent_labels, config.num_normal_labels)
        nn.init.xavier_uniform_(self.G)
        self.G = self.G.to(config.device)
    
    def forward(self, x, attention_mask=None, labels=None, labels_normal=None, lm_labels=None, labels_sent=None, labels_op=None):
        # x : (bsz, seq, hidden_size)
        o_normal = self.softmax(self.fc_normal(x)) # (bsz, seq, num_normal_labels)
        o_sent = self.softmax(self.fc_sent(x)) # (bsz, seq, num_sent_labels)

        # Loss computation
        loss = 0
        active_loss = attention_mask.view(-1) == 1

        # Boundary tag predictions
        active_logits = o_normal.view(-1, self.config.num_normal_labels)[active_loss]
        active_labels = labels_normal.view(-1)[active_loss]
        loss += nn.NLLLoss()(torch.log(active_logits), active_labels)

        # Sentiment tag predictions
        active_logits = o_sent.view(-1, self.config.num_sent_labels)[active_loss]
        active_labels = labels_sent.view(-1)[active_loss]
        loss += nn.NLLLoss()(torch.log(active_logits), active_labels)

        o = self.softmax(self.fc(x))
        batch_size = o.size()[0]
        seq_len = o.size()[1]

        o_sent = o_sent.unsqueeze(3) # (bsz, seq, num_sent_labels, 1)
        o_sent = o_sent.unsqueeze(4) # (bsz, seq, num_sent_labels, 1, 1)
        o_sent = o_sent.transpose(2, 4) # (bsz, seq, 1, 1, num_sent_labels)
        o_normal = o_normal.unsqueeze(3) # (bsz, seq, num_normal_labels, 1)
        o_normal = o_normal.unsqueeze(4) # (bsz, seq, num_normal_labels, 1, 1)
        o_normal = o_normal.transpose(2, 3) # (bsz, seq, 1, num_normal_labels, 1)

        o_prime = o_sent @ self.G @ o_normal
        o_prime = o_prime.squeeze()
        o_prime = self.softmax(o_prime)

        # compute confidence values
        alpha = self.epsilon * torch.sum(o_prime ** 2, dim=2)

        o_prime = o_prime.transpose(0, 1)
        o_prime = o_prime.transpose(0, 2) # (num_labels, bsz, seq)

        o = o.transpose(0, 1)
        o = o.transpose(0, 2) # (num_labels, bsz, seq)

        o_tilde = alpha * o_prime + (1 - alpha) * o

        o_tilde = o_tilde.transpose(0, 2)
        o_tilde = o_tilde.transpose(0, 1)

        # Final tag predictions
        active_logits = o_tilde.reshape(-1, self.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss += nn.NLLLoss()(torch.log(active_logits), active_labels)

        return loss, o_tilde

class BERTFCNCRF(nn.Module):

    def __init__(self, config):
        super(BERTFCNCRF, self).__init__()
        self.config = config
        self.epsilon = 0.5
        self.fc_normal = nn.Linear(config.hidden_size, config.num_normal_labels)
        self.fc_upper = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels)
        self.crf_normal = CRF(config.num_normal_labels)
        self.softmax = nn.Softmax(dim=2)

        # Transition matrix
        transition_path = {'B': ['B-POS', 'B-NEG', 'B-NEU'],
                           'I': ['I-POS', 'I-NEG', 'I-NEU'],
                           'O': ['O']}
        self.W_trans = torch.zeros(config.num_normal_labels, config.num_labels)
        for t in transition_path:
            next_tags = transition_path[t]
            n_next_tag = len(next_tags)
            ote_id = config.ote_vocab[t]
            for nt in next_tags:
                ts_id = config.ts_vocab[nt]
                self.W_trans[ote_id, ts_id] = 1.0 / n_next_tag
        self.W_trans = self.W_trans.to(config.device)
    
    def forward(self, x, attention_mask=None, labels=None, labels_normal=None, lm_labels=None, labels_sent=None, labels_op=None):
        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        h_t = x
        z_t = self.fc_normal(h_t)

        # Loss computation
        loss = 0
        active_loss = attention_mask.view(-1) == 1

        # Boundary tag predictions
        log_likelihood = self.crf_normal(inputs=z_t, tags=labels_normal, mask=attention_mask)
        loss += -log_likelihood

        # compute confidence values
        alpha = self.epsilon * torch.sum(self.softmax(z_t) ** 2, dim=2)

        # Unified tagging output
        z_s = self.fc_upper(h_t)

        z_s = z_s.transpose(0, 1)
        z_s = z_s.transpose(0, 2) # num_labels, bsz, seq

        # Transistion component
        z_s_prime = z_t @ self.W_trans # bsz, seq, num_labels

        z_s_prime = z_s_prime.transpose(0, 1)
        z_s_prime = z_s_prime.transpose(0, 2) # num_labels, bsz, seq

        # Combining both components
        z_s_tilde = z_s_prime * alpha + z_s * (1 - alpha)

        z_s_tilde = z_s_tilde.transpose(0, 2)
        z_s_tilde = z_s_tilde.transpose(0, 1) # bsz, seq, num_labels

        # Unified tagging
        log_likelihood = self.crf(inputs=z_s_tilde, tags=labels, mask=attention_mask)
        loss += -log_likelihood

        return loss, z_s_tilde


class BERTFCN(nn.Module):

    def __init__(self, config):
        super(BERTFCN, self).__init__()
        self.config = config
        self.epsilon = 0.5
        self.fc_normal = nn.Linear(config.hidden_size, config.num_normal_labels)
        self.fc_upper = nn.Linear(config.hidden_size, config.num_labels)
        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(0)

        # Transition matrix
        transition_path = {'B': ['B-POS', 'B-NEG', 'B-NEU'],
                           'I': ['I-POS', 'I-NEG', 'I-NEU'],
                           'O': ['O']}
        self.W_trans = torch.zeros(config.num_normal_labels, config.num_labels)
        for t in transition_path:
            next_tags = transition_path[t]
            n_next_tag = len(next_tags)
            ote_id = config.ote_vocab[t]
            for nt in next_tags:
                ts_id = config.ts_vocab[nt]
                self.W_trans[ote_id, ts_id] = 1.0 / n_next_tag
        self.W_trans = self.W_trans.to(config.device)
    
    def forward(self, x, attention_mask=None, labels=None, labels_normal=None, lm_labels=None, labels_sent=None, labels_op=None):
        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        h_t = x
        z_t = self.softmax(self.dropout(self.fc_normal(h_t)))

        # Loss computation
        loss = 0
        active_loss = attention_mask.view(-1) == 1

        # Boundary tag predictions
        active_logits = z_t.view(-1, self.config.num_normal_labels)[active_loss]
        active_labels = labels_normal.view(-1)[active_loss]
        loss += nn.NLLLoss()(torch.log(active_logits), active_labels)

        # compute confidence values
        alpha = self.epsilon * torch.sum(z_t ** 2, dim=2)

        # Unified tagging output
        z_s = self.softmax(self.dropout(self.fc_upper(h_t)))

        z_s = z_s.transpose(0, 1)
        z_s = z_s.transpose(0, 2) # num_labels, bsz, seq

        # Transistion component
        z_s_prime = z_t @ self.W_trans # bsz, seq, num_labels

        z_s_prime = z_s_prime.transpose(0, 1)
        z_s_prime = z_s_prime.transpose(0, 2) # num_labels, bsz, seq

        # Combining both components
        z_s_tilde = z_s_prime * alpha + z_s * (1 - alpha)

        z_s_tilde = z_s_tilde.transpose(0, 2)
        z_s_tilde = z_s_tilde.transpose(0, 1) # bsz, seq, num_labels

        # Unified tagging
        active_logits = z_s_tilde.view(-1, self.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss += nn.NLLLoss()(torch.log(active_logits), active_labels)
        
        return loss, z_s_tilde

class BERTBiLSTM(nn.Module):

    def __init__(self, config):
        super(BERTBiLSTM, self).__init__()
        self.config = config
        self.dim_h = 384
        self.epsilon = 0.5
        self.fc_normal = nn.Linear(config.hidden_size, config.num_normal_labels)
        self.fc_opinion_1 = nn.Linear(config.hidden_size, 50)
        self.fc_opinion_2 = nn.Linear(50, 2) # 0 or 1 (target or non-target word)
        self.softmax = nn.Softmax(dim=2)
        self.lstm_upper = nn.LSTM(
            config.hidden_size, 
            self.dim_h, 
            bidirectional=True)
        self.lstm_dropout = nn.Dropout(0.1)
        self.fc_upper = nn.Linear(2 * self.dim_h, config.num_labels)
        self.fc_tmp = nn.Linear(config.hidden_size, 2 * self.dim_h)

        transition_path = {'B': ['B-POS', 'B-NEG', 'B-NEU'],
                           'I': ['I-POS', 'I-NEG', 'I-NEU'],
                           'O': ['O']}
        self.W_trans = torch.zeros(config.num_normal_labels, config.num_labels)
        for t in transition_path:
            next_tags = transition_path[t]
            n_next_tag = len(next_tags)
            ote_id = config.ote_vocab[t]
            for nt in next_tags:
                ts_id = config.ts_vocab[nt]
                self.W_trans[ote_id, ts_id] = 1.0 / n_next_tag
        self.W_trans = self.W_trans.to(config.device)
        self.W_gate = torch.empty(2 * self.dim_h, 2 * self.dim_h)
        nn.init.uniform_(self.W_gate, -0.2, 0.2)
        self.W_gate = self.W_gate.to(config.device)
    
    def forward(self, x, attention_mask=None, labels=None, labels_normal=None, lm_labels=None, labels_sent=None, labels_op=None):
        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        h_t = x
        z_t = self.fc_normal(h_t)
        # z_o = self.dropout(F.relu(self.fc_opinion_1(h_t)))
        # z_o = self.softmax(self.fc_opinion_2(z_o))

        # Loss computation
        loss = 0
        active_loss = attention_mask.view(-1) == 1

        # Boundary tag predictions
        active_logits = z_t.view(-1, self.config.num_normal_labels)[active_loss]
        active_labels = labels_normal.view(-1)[active_loss]
        loss += nn.CrossEntropyLoss()(active_logits, active_labels)

        # Opinion Enhancement
        # active_logits = z_o.view(-1, 2)[active_loss]
        # active_labels = lm_labels.view(-1)[active_loss]
        # loss += nn.NLLLoss()(torch.log(active_logits), active_labels)

        h_t = h_t.transpose(0, 1)
        h_s, _ = self.lstm_upper(h_t) # seq, bz, 2 * self.dim_h
        h_t = h_t.transpose(0, 1)
        
        seq_len = h_s.size()[0]
        h_prev = None
        h_s_tilde = []
        # Sentiment consistency
        for t in range(seq_len):
            if t == 0:
                h_curr = h_s[t]
            else:
                gt = torch.sigmoid(torch.matmul(h_s[t], self.W_gate))
                h_curr = gt * h_s[t] + (1 - gt) * h_prev
            h_prev = h_curr
            h_s_tilde.append(h_curr.view(1, -1, 2 * self.dim_h))

        h_s_tilde = torch.cat(h_s_tilde, dim=0) # seq, bsz, 2 * self.dim_h
        h_s_tilde = h_s_tilde.transpose(0, 1) # bsz, seq, 2 * self.dim_h
        h_s_tilde = self.lstm_dropout(h_s_tilde)

        z_s = self.fc_upper(h_s_tilde)
        alpha = self.epsilon * torch.sum(self.softmax(z_t) ** 2, dim=2)
        z_s = z_s.transpose(0, 1)
        z_s = z_s.transpose(0, 2) # num_labels, bsz, seq

        z_s_prime = z_t @ self.W_trans # bsz, seq, num_labels
        z_s_prime = z_s_prime.transpose(0, 1)
        z_s_prime = z_s_prime.transpose(0, 2) # num_labels, bsz, seq

        # Boundary guidance using transition matrix
        z_s_tilde = z_s_prime * alpha + z_s * (1 - alpha)
        z_s_tilde = z_s_tilde.transpose(0, 2)
        z_s_tilde = z_s_tilde.transpose(0, 1) # bsz, seq, num_labels

        # Unified tagging
        active_logits = z_s_tilde.view(-1, self.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss += nn.CrossEntropyLoss()(active_logits, active_labels)
        
        return loss, z_s_tilde


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
        self.tagger = CollaborativeLM(bert_config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, labels_normal=None, lm_labels=None,
                labels_sent=None, labels_op=None, idxs=None):
        outputs = self.bert(
            input_ids, 
            position_ids=position_ids, 
            token_type_ids=token_type_ids,
            attention_mask=attention_mask, 
            head_mask=head_mask)
        embeddings = outputs[2][11] # using output of 11th layer (last-second)
        x = self.bert_dropout(embeddings)
        loss, logits = self.tagger(x, attention_mask, labels, labels_normal, lm_labels, labels_sent, labels_op)
        outputs = (loss, logits,) + outputs[2:]
        return outputs
