from transformers import BertForMaskedLM, BertConfig
import torch
from cos_sim import CosSim
from use import USE
from utils import convert_to_batch, convert_to_dataset
from filter_words import filter_words

class Adversary:
    def __init__(self, args, tgt_model, k=10):
        self.args = args
        # Initialise BERT-MLM
        config_mlm = BertConfig.from_pretrained(args.model_name_or_path)
        self.mlm_model = BertForMaskedLM.from_pretrained(args.model_name_or_path, config=config_mlm, cache_dir='./cache').to(args.device)
        self.tgt_model = tgt_model
        # Build vocab for counter fitting embedding
        idx2word = {}
        word2idx = {}
        with open(args.counter_fitting_embedding_path, 'r') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in idx2word:
                    idx2word[len(idx2word)] = word
                    word2idx[word] = len(idx2word) - 1
        self.idx2word = idx2word
        self.word2idx = word2idx
        self.csim = CosSim(args.counter_fitting_embedding_path)
        # Universal sentence encoder
        self.use = USE()
        self.k = k
    
    def generate_adv_examples(self, examples, imp_words, tokenizer):
        model = self.tgt_model
        mlm_model = self.mlm_model
        args = self.args
        use = self.use
        batch = convert_to_batch(args, examples, tokenizer)
        model.eval()
        curr_outputs = model(**batch)
        curr_logits = curr_outputs[1]
        orig_preds = curr_logits.argmax(dim=-1)
        true_labels = batch['labels']
        adv_examples = []
        word_predictions = mlm_model(batch['input_ids'])[0] # (bsz, seq_len, vocab)
        word_scores, word_ids = torch.topk(word_predictions, self.k, -1) # (bsz, seq_len, k)

        success = 0
        for i, (example, positions) in enumerate(zip(examples, imp_words)):
            match_ratio = self.find_match_ratio(true_labels[i], orig_preds[i], args.device).item()
            if match_ratio < 0.5:
                adv_examples.append(example)
                success += 1
                continue
            orig = example.text_a.split(' ')
            words = orig.copy()
            for tid, wid in positions:
                word = words[wid].lower()
                # get replacement candidates
                sim_words, scores = self.get_substitues(
                    word_ids[i, tid, :], 
                    word_scores[i, tid, :], 
                    tokenizer, 
                    word)
                if len(sim_words) == 0:
                    continue
                sim_words = sim_words.to(args.device)
                scores = scores.to(args.device)
                new_sentences = []
                orig_sentences = []
                new_examples = []
                for new_word in sim_words:
                    orig_sentences.append(' '.join(orig))
                    new_sentence = words.copy()
                    new_sentence[wid] = tokenizer.convert_ids_to_tokens(new_word.item())
                    sentence = ' '.join(new_sentence)
                    new_sentences.append(sentence)

                    new_example = example.copy()
                    new_example.text_a = sentence
                    new_examples.append(new_example)
                scores = use.semantic_sim(orig_sentences, new_sentences)
                scores = torch.tensor(scores).to(args.device)
                # ensure semantic similarity
                mask = scores > 0.8
                scores = scores[mask]
                new_data = convert_to_batch(args, new_examples, tokenizer)
                for key in new_data:
                    new_data[key] = new_data[key][mask]
                sim_words = sim_words[mask]
                if len(new_data['input_ids']) == 0:
                    continue
                model.eval()
                # Evaluate new examples on target model
                new_data_outputs = model(**new_data)
                logits = new_data_outputs[1]
                logits = torch.nn.Softmax(dim=-1)(logits)
                new_preds = logits.argmax(dim=-1)
                # Check if accuracy is less than 0.5, if yes then terminate with example (for faster running time)
                match_ratios = self.find_match_ratio(new_data['labels'], new_preds, args.device)
                cond = match_ratios < 0.5
                if cond.any():
                    sim_words = sim_words[cond]
                    scores = scores[cond]
                    idx = scores.argmax(dim=-1).item()
                    selected_word = sim_words[idx].item()
                    words[wid] = tokenizer.convert_ids_to_tokens(selected_word)
                    success += 1
                    break
                else:
                    tl = new_data['labels'] # (bsz, seq_len)
                    mask = (new_data['labels'] > 0).type(torch.float)
                    logits = logits.gather(2, tl.unsqueeze(2)).squeeze(2)
                    logits = logits * mask # (bsz, seq)
                    confidence = logits.sum(dim=-1)
                    idx = confidence.argmin(dim=-1).item()
                    selected_word = sim_words[idx].item()
                    words[wid] = tokenizer.convert_ids_to_tokens(selected_word)
            adv_example = example.copy()
            adv_example.text_a = ' '.join(words)
            adv_examples.append(adv_example)
        return adv_examples
    
    def find_match_ratio(self, true_label, pred_label, device):
        mask = true_label > 0
        mask = mask.type(torch.float)
        tot = mask.sum(dim=-1)
        tot = torch.max(tot, torch.tensor(1., dtype=torch.float).to(device))
        
        true_label = true_label * mask
        pred_label = pred_label * mask
        diff = (true_label != pred_label)
        diff = diff.type(torch.float).sum(dim=-1)
        diff = 1 - diff / tot
        return diff
    
    def get_substitues(self, word_ids, word_scores, tokenizer, tgt_word):
        words = []
        scores = []
        for (wid, score) in zip(word_ids, word_scores):
            word = tokenizer.convert_ids_to_tokens(wid.item()).lower()
            # Ignore BERT subwords for now
            if '##' in word:
                continue
            if word in filter_words:
                continue
            if word in self.word2idx and tgt_word in self.word2idx:
                if self.csim.compute_sim(self.word2idx[word], self.word2idx[tgt_word]) < 0.4:
                    continue
            words.append(wid)
            scores.append(score)
        return torch.tensor(words), torch.tensor(scores)