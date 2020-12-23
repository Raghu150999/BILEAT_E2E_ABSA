# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team and authors from University of Illinois at Chicago.
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


import os
import logging
import argparse
import random
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import get_linear_schedule_with_warmup, AdamW
from transformers import BertModel, PreTrainedModel, BertForPreTraining, BertConfig

import modelconfig
from tqdm import tqdm


class BertForMTPostTraining(nn.Module):
    def __init__(self, model, config):
        super(BertForMTPostTraining, self).__init__()
        self.model = model
        self.bert = self.model.bert
        self.config = config

    def KL(self, adv, orig, mask):
        loss = F.kl_div(
            F.log_softmax(adv[0].view(-1, self.config.vocab_size)[mask], dim=-1, dtype=torch.float32), 
            F.softmax(orig[0].view(-1, self.config.vocab_size)[mask], dim=-1, dtype=torch.float32))
        loss += F.kl_div(
            F.log_softmax(adv[1].view(-1, 2), dim=-1, dtype=torch.float32), 
            F.softmax(orig[1].view(-1, 2), dim=-1, dtype=torch.float32))
        return loss

    def compute_loss(self, logits, labels):
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        return loss_fct(logits, labels)
    
    def get_loss(self, logits, labels, mask):
        loss = 0
        loss += self.compute_loss(
            logits[0].view(-1, self.config.vocab_size)[mask], 
            labels[0].view(-1)[mask])
        loss += self.compute_loss(
            logits[1].view(-1, 2),
            labels[1].view(-1))
        return loss

    def adv_project(self, grad, eps=1e-6):
        direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction

    def compute_logits(self, input_ids=None, token_type_ids=None, attention_mask=None, inputs_embeds=None):
        if inputs_embeds != None:
            return self.model(token_type_ids=token_type_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds[input_ids])
        else:
            return self.model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)


    def forward(self, mode, input_ids=None, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        # Implementation of Algorithm 1
        mask = attention_mask.view(-1) == 1
        embed = self.bert.embeddings.word_embeddings.weight
        noise_var = 1e-5
        noise = torch.randn(embed.size()).to('cuda') * noise_var # Line 3
        noise.requires_grad_(True)
        newembed = embed.data.detach() + noise
        logits1, logits2 = self.compute_logits(
            input_ids=input_ids, token_type_ids=token_type_ids,
            attention_mask=attention_mask)
        loss = self.get_loss(
            [logits1, logits2],
            [masked_lm_labels, next_sentence_label],
            mask)

        adv_logits1, adv_logits2 = self.compute_logits(
            input_ids=input_ids, token_type_ids=token_type_ids,
            attention_mask=attention_mask, inputs_embeds=newembed)
        adv_loss = self.KL([adv_logits1, adv_logits2], [logits1.detach(), logits2.detach()], mask)
        
        delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True) # Line 5
        norm = delta_grad.norm()

        if (torch.isnan(norm) or torch.isinf(norm)):
            # skip this batch
            print('Batch skipped')
            return loss

        eta = 1e-4
        noise = noise + delta_grad * eta # Line 6
        noise = self.adv_project(noise) # Line 6

        newembed = embed.data.detach() + noise
        newembed = newembed.detach()

        adv_logits1, adv_logits2 = self.compute_logits(
            input_ids=input_ids, token_type_ids=token_type_ids,
            attention_mask=attention_mask, inputs_embeds=newembed)

        adv_loss_f = self.KL([adv_logits1, adv_logits2], [logits1.detach(), logits2.detach()], mask)
        adv_loss_b = self.KL([logits1.detach(), logits2.detach()], [adv_logits1, adv_logits2], mask)
        
        alpha = 0.5
        adv_loss = (adv_loss_f + adv_loss_b) * alpha # Line 8
        loss += adv_loss # Line 8
        return loss

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def train(args):
    #load squad data for pre-training.
    
    args.train_batch_size=int(args.train_batch_size / args.gradient_accumulation_steps)
    
    review_train_examples=np.load(os.path.join(args.review_data_dir, "data.npz") )
    
    num_train_steps = args.num_train_steps
    bar = tqdm(total=num_train_steps)
    
    # load bert pre-train data.
    review_train_data = TensorDataset(
        torch.from_numpy(review_train_examples["input_ids"]),
        torch.from_numpy(review_train_examples["segment_ids"]),
        torch.from_numpy(review_train_examples["input_mask"]),           
        torch.from_numpy(review_train_examples["masked_lm_ids"]),
        torch.from_numpy(review_train_examples["next_sentence_labels"]) )
    
    review_train_dataloader = DataLoader(review_train_data, sampler=RandomSampler(review_train_data), batch_size=args.train_batch_size , drop_last=True)
    
    # we do not have any valiation for pretuning
    model = BertForPreTraining.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model], cache_dir='../cache', config=BertConfig())
    model.train()
    model = BertForMTPostTraining(model, BertConfig())
    
    model.cuda()

    # Prepare optimizer
    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad==True]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps
        
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total)
    global_step=0
    step=0
    batch_loss=0.
    model.train()
    model.zero_grad()
    
    training=True
    
    review_iter=iter(review_train_dataloader)
    model_dir = os.path.join(args.output_dir, "saved_model")
    os.makedirs(model_dir, exist_ok=True)
    while training:
        try:
            batch = next(review_iter)
        except:
            review_iter=iter(review_train_dataloader)
            batch = next(review_iter)
            
        batch = tuple(t.cuda() for t in batch)
        
        input_ids, segment_ids, input_mask, masked_lm_ids, next_sentence_labels = batch
        
        review_loss = model("review", input_ids=input_ids.long(), token_type_ids=segment_ids.long(), attention_mask=input_mask.long(), masked_lm_labels=masked_lm_ids.long(), next_sentence_label=next_sentence_labels.long())
        
        loss = review_loss

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        batch_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            bar.update(1)
            if global_step % 50 ==0:
                logging.info("step %d batch_loss %f ", global_step, batch_loss)
            batch_loss=0.

            if global_step % args.save_checkpoints_steps == 0:
                model.float()
                print('Saving model..')
                model.model.save_pretrained(model_dir + f"-{global_step}")
            if global_step>=num_train_steps:
                training=False
                break
        step+=1
    model.float()
    print('Saving model..')
    model.model.save_pretrained(model_dir + f"-{global_step}")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model", default="bert-base", type=str, required=True, help="pretrained weights of bert.")

    parser.add_argument("--review_data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="dir of review numpy file dir.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--train_batch_size", default=16,
                        type=int, help="training batch size for both review and squad.")
        
    parser.add_argument("--do_train", default=False, action="store_true", help="Whether to run training.")

    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")

    parser.add_argument("--num_train_steps", default=50000,
                        type=int, help="Number of training steps.")

    parser.add_argument("--warmup_proportion", default=0.1,
                        type=float, help="Number of warmup steps.")

    parser.add_argument("--save_checkpoints_steps", default=1000,
                        type=int, help="How often to save the model checkpoint.")

    parser.add_argument("--seed", default=12345,
                        type=int, help="random seed.")

    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=2)

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)        
        train(args)

if __name__ == "__main__":
    main()
