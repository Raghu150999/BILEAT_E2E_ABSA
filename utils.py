from glue_utils import ABSAProcessor, convert_examples_to_seq_features
import torch
from dataset import ABSADataset


def convert_to_dataset(args, examples, tokenizer):
    processor = ABSAProcessor()
    label_list = processor.get_labels(args.tagging_schema)
    normal_labels = processor.get_normal_labels(args.tagging_schema)
    features, imp_words = convert_examples_to_seq_features(
        examples=examples, label_list=(label_list, normal_labels),
        tokenizer=tokenizer,
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        sep_token=tokenizer.sep_token,
        cls_token_segment_id=0,
        pad_on_left=False,
        pad_token_segment_id=0)
    idxs = torch.arange(len(features))
    dataset = ABSADataset(features, idxs)
    return dataset

def convert_to_batch(args, examples, tokenizer):
    dataset = convert_to_dataset(args, examples, tokenizer)
    batch = dataset[:]
    for key in batch:
        batch[key] = batch[key].to(args.device)
    return batch
