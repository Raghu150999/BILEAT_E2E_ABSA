import argparse
import os
import torch
import logging
import random
import numpy as np
import shutil

from glue_utils import convert_examples_to_seq_features, output_modes, processors, compute_metrics_absa, ABSAProcessor
from tqdm import tqdm, trange
from transformers import BertConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from absa_layer import BertABSATagger
from dataset import ABSADataset
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from adversary import Adversary
from utils import convert_to_batch, convert_to_dataset

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertABSATagger, AutoTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--absa_type", default=None, type=str, required=True,
                        help="Downstream absa layer type selected in the list: [linear, gru, san, tfm, crf]")
    parser.add_argument("--tfm_mode", default=None, type=str, required=True,
                        help="mode of the pre-trained transformer, selected from: [finetune]")
    parser.add_argument("--fix_tfm", default=None, type=int, required=True,
                        help="whether fix the transformer params or not")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--counter_fitting_embedding_path", default="./counter-fitted-vectors.txt", type=str, help="Path to counter fitting embeddings for cosine similarity")
    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    
    # Adversarial training args
    parser.add_argument("--do_adv", action='store_true',
                        help="Whether to perform adverserial training or not")
    parser.add_argument("--gen_adv_from_path", type=str, default='',
                        help="Generates adversarial data from model loaded from path")
    parser.add_argument("--adv_data_path", default='', type=str, help="Loads dataset from pretrained path")
    parser.add_argument("--adv_loss_weight", default=0.5, type=float, help="Lambda for adverserial loss")
    parser.add_argument("--pred_checkpoint", default='', type=str, help="Generate predictions for checkpoint")
    parser.add_argument("--load_model", default='', type=str, help="Loads model from path")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    parser.add_argument('--tagging_schema', type=str, default='BIEOS')

    parser.add_argument("--overfit", type=int, default=0,
                        help="if evaluate overfit or not")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='',
                        help="For distant debugging.")
    parser.add_argument('--server_port', type=str,
                        default='', help="For distant debugging.")
    parser.add_argument('--MASTER_ADDR', type=str)
    parser.add_argument('--MASTER_PORT', type=str)
    args = parser.parse_args()
    output_dir = '%s-%s' % (args.absa_type, args.task_name)

    if args.fix_tfm:
        output_dir = '%s-fix' % output_dir
    if args.overfit:
        output_dir = '%s-overfit' % output_dir
        args.max_steps = 3000
    args.output_dir = output_dir
    return args

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # draw training samples from shuffled dataset
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    if args.do_adv:
        adv_dataset = torch.load(args.adv_data_path)
        adv_sampler = RandomSampler(adv_dataset)
        adv_dataloader = DataLoader(adv_dataset, sampler=adv_sampler, batch_size=args.train_batch_size)
    else:
        adv_dataloader = train_dataloader
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    loss_file = open(f'{args.output_dir}/loss.txt', 'a')
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=False)
    # Set the seed number
    # Added here for reproductibility
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(zip(train_dataloader, adv_dataloader), desc="Iteration", disable=False)
        for step, (train_batch, adv_batch) in enumerate(epoch_iterator):            
            for key in train_batch:
                train_batch[key] = train_batch[key].to(args.device)
            model.train()
            outputs = model(**train_batch)
            # loss with attention mask
            # model outputs are always tuple in pytorch-transformers (see doc)
            loss = outputs[0]
            if args.do_adv:
                for key in adv_batch:
                    adv_batch[key] = adv_batch[key].to(args.device)
                outputs_adv = model(**adv_batch)
                loss_adv = outputs_adv[0]
                loss = loss + args.adv_loss_weight * loss_adv
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not average well
                    tr_loss_cp = (tr_loss - logging_loss) / args.logging_steps
                    loss_file.write(f'Step: {global_step}, Training loss: {tr_loss_cp}\n')
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, mode='dev', prefix=global_step)
                        loss_file.write(f'Step: {global_step}, Validation loss: {results["eval_loss"]}, ')
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                'eval_{}'.format(key), value, global_step)
                            loss_file.write(f'{key}: {value}, ')
                        loss_file.write("\n")
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    logger.info(f"\nTrain loss: {tr_loss_cp}\n")
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint per each N steps
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(
                        output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    tb_writer.close()
    loss_file.write('-' * 60)
    loss_file.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, mode, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, eval_evaluate_label_ids, examples, _ = load_and_cache_examples(
            args, eval_task, tokenizer, mode=mode)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * \
            max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(
            eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        # logger.info("***** Running evaluation on %s.txt *****" % mode)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        crf_logits, crf_mask = [], []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            with torch.no_grad():
                inputs = batch
                for key in inputs:
                    inputs[key] = inputs[key].to(args.device)
                outputs = model(**inputs)
                # logits: (bsz, seq_len, label_size)
                # here the loss is the masked loss
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()

                crf_logits.append(logits)
                crf_mask.append(batch['attention_mask'])
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
        # argmax operation over the last dimension
        if model.tagger_config.absa_type[-3:] != 'crf':
            # greedy decoding
            preds = np.argmax(preds, axis=-1)
            print(np.max(preds))
        else:
            # viterbi decoding for CRF-based model
            crf_logits = torch.cat(crf_logits, dim=0)
            crf_mask = torch.cat(crf_mask, dim=0)
            preds = model.tagger.crf.viterbi_tags(logits=crf_logits, mask=crf_mask)
        result, tagging = compute_metrics_absa(
            preds, out_label_ids, eval_evaluate_label_ids, args.tagging_schema)
        result['eval_loss'] = eval_loss
        results.update(result)
        if mode == 'test':
            qpreds = open(f'{args.output_dir}/qpred.txt', 'w')
            for example, tags in zip(examples, tagging):
                qpreds.write(example.text_a)
                qpreds.write("\n")
                qpreds.write("True labels: \n")
                for (b, e, s) in tags[0]:
                    qpreds.write(f'{b} {e} {s}, ')
                qpreds.write("\n")
                qpreds.write("Predicted labels: \n")
                for (b, e, s) in tags[1]:
                    qpreds.write(f'{b} {e} {s}, ')
                qpreds.write("\n")
            qpreds.close()

        output_eval_file = os.path.join(eval_output_dir, "%s_results.txt" % mode)
        with open(output_eval_file, "w") as writer:
            #logger.info("***** %s results *****" % mode)
            for key in sorted(result.keys()):
                if 'eval_loss' in key:
                    logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
            #logger.info("***** %s results *****" % mode)
    return results

def load_and_cache_examples(args, task, tokenizer, mode='train', model=None):
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        print("cached_features_file:", cached_features_file)
        data = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels(args.tagging_schema)
        normal_labels = processor.get_normal_labels(args.tagging_schema)
        if mode == 'train':
            examples = processor.get_train_examples(
                args.data_dir, args.tagging_schema)
        elif mode == 'dev':
            examples = processor.get_dev_examples(
                args.data_dir, args.tagging_schema)
        elif mode == 'test':
            examples = processor.get_test_examples(
                args.data_dir, args.tagging_schema)
        else:
            raise Exception("Invalid data mode %s..." % mode)
        data = convert_examples_to_seq_features(
            examples=examples, label_list=(label_list, normal_labels),
            tokenizer=tokenizer,
            cls_token_at_end=False,
            cls_token=tokenizer.cls_token,
            sep_token=tokenizer.sep_token,
            cls_token_segment_id=0,
            pad_on_left=False,
            pad_token_segment_id=0)
        data = data + (examples,)
        torch.save(data, cached_features_file)
    features, imp_words, examples = data
    all_evaluate_label_ids = [f.evaluate_label_ids for f in features]
    idxs = torch.arange(len(features))
    dataset = ABSADataset(features, idxs)
    return dataset, all_evaluate_label_ids, examples, imp_words


def main():
    args = init_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        os.environ['MASTER_ADDR'] = args.MASTER_ADDR
        os.environ['MASTER_PORT'] = args.MASTER_PORT
        torch.distributed.init_process_group(
            backend='nccl', rank=args.local_rank, world_size=1)
        args.n_gpu = 1

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # not using 16-bits training
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: False", args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % args.task_name)
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels(args.tagging_schema)
    num_labels = len(label_list)
    normal_labels = processor.get_normal_labels(args.tagging_schema)
    num_normal_labels = len(normal_labels)
    sent_labels = ABSAProcessor.get_sentiment_labels()
    num_sent_labels = len(sent_labels)

    # initialize the pre-trained model
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name)
    tokenizer =  AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir='./cache')

    config.absa_type = args.absa_type
    config.tfm_mode = args.tfm_mode
    config.fix_tfm = args.fix_tfm
    config.num_normal_labels = num_normal_labels
    config.num_sent_labels = num_sent_labels
    config.ts_vocab = {label : i for i, label in enumerate(label_list)}
    config.ote_vocab = {label : i for i, label in enumerate(normal_labels)}
    config.sent_vocab = {label : i for i, label in enumerate(sent_labels)}
    config.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    config.output_hidden_states = True
    config.model_name_or_path = args.model_name_or_path

    if args.gen_adv_from_path:
        # Generate adversarial examples
        modes = ['train', 'dev', 'test']
        for mode in modes:
            model = model_class.from_pretrained(args.gen_adv_from_path).to(args.device)
            train_dataset, train_evaluate_label_ids, examples, imp_words = load_and_cache_examples(
                args, args.task_name, tokenizer, mode=mode, model=model)
            adversary = Adversary(args, model)
            adv_examples = []
            sz = 64
            for _ in trange(len(examples) // sz + 1):
                if len(examples) == 0:
                    continue
                adv_examples.extend(adversary.generate_adv_examples(examples[:sz], imp_words[:sz], tokenizer))
                examples = examples[sz:]
                imp_words = imp_words[sz:]
            adv_dataset = convert_to_dataset(args, adv_examples, tokenizer)
            output_dir = f'{args.task_name}_adv'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            torch.save(adv_dataset, f'{output_dir}/{mode}.pth')
            torch.save(adv_examples, f'{output_dir}/{mode}-examples.pth')
        exit(0)
    
    if args.load_model:
        print('Loading model from:', args.load_model)
        model = model_class.from_pretrained(args.load_model, config=config)
    else:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            config=config,
            cache_dir='./cache')
        print('Loading model from:', args.model_name_or_path)

    # Distributed and parallel training
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Training
    if args.do_train:
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.mkdir(args.output_dir)

        # Store model configuration with results
        shutil.copyfile('absa_layer.py', args.output_dir + '/absa_layer.py')
        # Store training configuration
        shutil.copyfile('train.sh', args.output_dir + '/train.sh')
        if args.do_adv:
            # Store adv training config
            shutil.copyfile('main.py', args.output_dir + '/main.py')
        
        train_dataset, train_evaluate_label_ids, examples, imp_words = load_and_cache_examples(
            args, args.task_name, tokenizer, mode='train', model=model)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        # save the model configuration
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Validation
    results = {}
    best_f1 = -999999.0
    best_checkpoint = None
    checkpoints = []
    if args.eval_all_checkpoints:
        checkpoints = os.listdir(args.output_dir)
        checkpoints.sort()
    logger.info(
        "Perform validation on the following checkpoints: %s", checkpoints)
    test_results = {}
    steps = []
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        if checkpoint.split('-')[0] != 'checkpoint':
            continue
        if args.pred_checkpoint and args.pred_checkpoint != global_step:
            continue
        steps.append(global_step)
        set_seed(args)
        model = model_class.from_pretrained(f'{args.output_dir}/{checkpoint}')
        model.to(args.device)
        dev_result = evaluate(args, model, tokenizer,
                              mode='dev', prefix=global_step)

        # regard the micro-f1 as the criteria of model selection
        if int(global_step) > 1000 and dev_result['micro-f1'] > best_f1:
            best_f1 = dev_result['micro-f1']
            best_checkpoint = checkpoint
        dev_result = dict((k + '_{}'.format(global_step), v)
                        for k, v in dev_result.items())
        results.update(dev_result)

        test_result = evaluate(args, model, tokenizer,
                            mode='test', prefix=global_step)
        test_result = dict((k + '_{}'.format(global_step), v)
                        for k, v in test_result.items())
        test_results.update(test_result)

    best_ckpt_string = "\nThe best checkpoint is %s" % best_checkpoint
    logger.info(best_ckpt_string)
    dev_f1_values, dev_loss_values = [], []
    for k in results:
        v = results[k]
        if 'micro-f1' in k:
            dev_f1_values.append((k, v))
        if 'eval_loss' in k:
            dev_loss_values.append((k, v))
    test_f1_values, test_loss_values = [], []
    for k in test_results:
        v = test_results[k]
        if 'micro-f1' in k:
            test_f1_values.append((k, v))
        if 'eval_loss' in k:
            test_loss_values.append((k, v))
    log_file_path = '%s/log.txt' % args.output_dir
    log_file = open(log_file_path, 'a')
    log_file.write("\tValidation:\n")
    for (test_f1_k, test_f1_v), (test_loss_k, test_loss_v), (dev_f1_k, dev_f1_v), (dev_loss_k, dev_loss_v) in zip(
            test_f1_values, test_loss_values, dev_f1_values, dev_loss_values):
        global_step = int(test_f1_k.split('_')[-1])
        if not args.overfit and global_step <= 1000:
            continue
        print('test-%s: %.5lf, test-%s: %.5lf, dev-%s: %.5lf, dev-%s: %.5lf' % (test_f1_k,
                                                                                test_f1_v, test_loss_k, test_loss_v,
                                                                                dev_f1_k, dev_f1_v, dev_loss_k,
                                                                                dev_loss_v))
        validation_string = '\t\tdev-%s: %.5lf, dev-%s: %.5lf' % (
            dev_f1_k, dev_f1_v, dev_loss_k, dev_loss_v)
        log_file.write(validation_string+'\n')

    n_times = args.max_steps // args.save_steps + 1
    for step in steps:
        log_file.write('\tStep %s:\n' % step)
        precision = test_results['precision_%s' % step]
        recall = test_results['recall_%s' % step]
        micro_f1 = test_results['micro-f1_%s' % step]
        macro_f1 = test_results['macro-f1_%s' % step]
        log_file.write('\t\tprecision: %.4lf, recall: %.4lf, micro-f1: %.4lf, macro-f1: %.4lf\n'
                       % (precision, recall, micro_f1, macro_f1))
    log_file.write("\tBest checkpoint: %s\n" % best_checkpoint)
    log_file.write('******************************************\n')
    log_file.close()


if __name__ == '__main__':
    main()
