import torch
from torch.utils.data import Dataset

class ABSADataset(Dataset):

    def __init__(self, features, idxs):
        # Convert to Tensors and build dataset
        self.all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        self.all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        self.all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
        self.all_label_ids_1 = torch.tensor([f.label_ids_1 for f in features], dtype=torch.long)
        self.all_label_ids_o = torch.tensor([f.label_ids_o for f in features], dtype=torch.long)
        self.stm_lm_labels = torch.tensor([f.stm_lm_labels for f in features], dtype=torch.long)
        self.labels_sent = torch.tensor([f.label_sent for f in features], dtype=torch.long)
        self.idxs = idxs
    
    def __len__(self):
        return len(self.all_input_ids)
    
    def __getitem__(self, idx):
        data = {
            'input_ids':      self.all_input_ids[idx],
            'attention_mask': self.all_input_mask[idx],
            'token_type_ids': self.all_segment_ids[idx],
            'labels':         self.all_label_ids[idx],
            'labels_normal':  self.all_label_ids_1[idx],
            'labels_op':      self.all_label_ids_o[idx],
            'lm_labels':      self.stm_lm_labels[idx],
            'labels_sent':    self.labels_sent[idx],
            'idxs':           self.idxs[idx]
            }
        return data