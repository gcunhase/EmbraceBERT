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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa).
    Copy of run_classifier on May 4th 2020
    
    1. Fine-tune BERT with complete sentences (BERTc)
    2. Use BERTc in 2 models: 
        Q = output of BERTc fine-tuned with complete sentences (frozen weights)
        K, V = output of BERT fine-tuned with complete sentences and further fine-tuned with incomplete sentences 
              (adjust weights) -> BERTi
        Q, K, V in multi-head attention followed by embracement layer, resulting in vector e. Attention layer applied 
                between CLS and e in order to classify text.
    3. During test, Q is also obtained from BERTi 
    
    Modification:
        - Save best model only when args.save_best = True
        - Write accuracy in JSON file
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import shutil
import json
from utils import ensure_dir

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME,
                                  BertConfig, BertTokenizer,
                                  RobertaConfig, RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer,
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer)

from models.BERT_Dropout import BertForSequenceClassification
from models.RoBERTa_Dropout import RobertaForSequenceClassification
from models.EmbraceBERT import EmbraceBertForSequenceClassification
from models.EmbraceBERTwithQuery import EmbraceBertWithQueryForSequenceClassification
from models.EmbraceRoBERTa import EmbraceRobertaForSequenceClassification

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_classifier import (compute_metrics, convert_examples_to_features,
                              output_modes, processors, labels_array)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'embracebert': (BertConfig, EmbraceBertForSequenceClassification, BertTokenizer),
    'embracebertwithquery': (BertConfig, EmbraceBertWithQueryForSequenceClassification , BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'embraceroberta': (RobertaConfig, EmbraceRobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def pre_train(args, train_dataset, model, tokenizer, min_loss=float("inf"), eval_data_type="train",
              freeze_bert_weights=False, train_bertc=False):
    """ Pre-train the model """

    output_dir = args.output_dir
    num_train_epochs = args.num_train_epochs
    log_dir = args.log_dir
    if train_bertc:
        output_dir = args.output_dir_complete
        num_train_epochs = args.num_train_epochs_bertc
        log_dir = args.log_dir_complete

    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger_additional_str = ""
    if freeze_bert_weights:
        logger_additional_str = " Frozen BERT "
    logger.info("***** Running training {}*****".format(logger_additional_str))
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['embracebert', 'bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            if args.model_type in ['embracebert', 'embraceroberta', 'bert', 'roberta']:
                outputs = model(**inputs, apply_dropout=args.apply_dropout, freeze_bert_weights=freeze_bert_weights)
            elif args.model_type in ['embracebertwithquery']:
                outputs = model(**inputs, apply_dropout=args.apply_dropout, freeze_bert_weights=freeze_bert_weights)
            else:
                outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # Save best model
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 and args.save_best:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training and global_step % args.evaluate_steps == 0:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, train_bertc=train_bertc)  #, data_type=eval_data_type)
                        # for key, value in results.items():
                        #    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        # Saving checkpoint by checking eval loss was too slow and memory expensive.
                        # if results['eval_loss'] < min_loss:
                        #     min_loss = results['eval_loss']
                        logger.info("Eval loss{}: {}".format(logger_additional_str, results['eval_loss']))
                        tb_writer.add_scalar('eval_loss', results['eval_loss'], global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar('logging_loss', logging_loss, global_step)
                    tb_writer.add_scalar('loss_item', loss.item(), global_step)
                    logger.info("Loss{}: {}".format(logger_additional_str, (tr_loss - logging_loss) / args.logging_steps))
                    logger.info("Logging loss{}: {}".format(logger_additional_str, logging_loss))
                    logger.info("Loss item{}: {}".format(logger_additional_str, loss.item()))
                    logger.info("Global step/Step{}: {}/{}".format(logger_additional_str, global_step, step))
                    logging_loss = tr_loss

                    # Save best checkpoint
                    if loss.item() < min_loss:
                        logger.info(
                            "Loss item - Previous: {}, Current min: {}, Global step: {}".format(min_loss, loss.item(),
                                                                                                global_step))
                        min_loss = loss.item()
                        # Save best model checkpoint
                        prefix = 'best'
                        if freeze_bert_weights:
                            prefix = 'pretrain'
                        # Find and remove last best
                        list_dirs = os.listdir(output_dir)
                        best_dir_ckpt = [s for s in list_dirs if prefix in s]
                        if len(best_dir_ckpt) > 0:
                            shutil.rmtree(os.path.join(output_dir, best_dir_ckpt[0]))
                        # Create new checkpoint directory
                        output_dir_ckp = os.path.join(output_dir, 'checkpoint-{}-{}'.format(prefix, global_step))
                        if not os.path.exists(output_dir_ckp):
                            os.makedirs(output_dir_ckp)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir_ckp)
                        torch.save(args, os.path.join(output_dir_ckp, 'training_args.bin'))
                        logger.info("Saving {}model checkpoint to {}".format(logger_additional_str, output_dir_ckp))

                # Save model checkpoint every X steps if save_best = False
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0 and not args.save_best:
                    output_dir_ckp = os.path.join(output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir_ckp):
                        os.makedirs(output_dir_ckp)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir_ckp)
                    torch.save(args, os.path.join(output_dir_ckp, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir_ckp)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def train(args, train_dataset_complete, train_dataset_incomplete, model_bertc, model, tokenizer, min_loss=float("inf"), eval_data_type="train"):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset_complete) if args.local_rank == -1 else DistributedSampler(train_dataset_complete)
    train_dataloader_complete = DataLoader(train_dataset_complete, sampler=train_sampler, batch_size=args.train_batch_size)
    train_dataloader_incomplete = DataLoader(train_dataset_incomplete, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader_complete) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader_complete) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader_complete))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator_comp = tqdm(train_dataloader_complete, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator = tqdm(train_dataloader_incomplete, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, (batch_comp, batch) in enumerate(zip(epoch_iterator_comp, epoch_iterator)):
            model.train()
            batch_comp = tuple(t.to(args.device) for t in batch_comp)
            inputs_comp = {'input_ids':      batch_comp[0],
                           'attention_mask': batch_comp[1],
                           'token_type_ids': batch_comp[2] if args.model_type in ['embracebert', 'bert', 'xlnet', 'embracebertwithquery'] else None,  # XLM and RoBERTa don't use segment_ids
                           'labels':         batch_comp[3]}
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['embracebert', 'bert', 'xlnet', 'embracebertwithquery'] else None,  # XLM and RoBERTa don't use segment_ids
                      'labels':         batch[3]}
            if args.model_type in ['embracebert', 'embraceroberta', 'bert', 'roberta']:
                outputs = model(**inputs, apply_dropout=args.apply_dropout)
            elif args.model_type in ['embracebertwithquery']:
                outputs = model(**inputs, input_bertc=inputs_comp, apply_dropout=args.apply_dropout, model_bertc=model_bertc)
            else:
                outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                # Save best model
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0 and args.save_best:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training and global_step % args.evaluate_steps == 0:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer, is_evaluate=True)  #, data_type=eval_data_type)
                        # for key, value in results.items():
                        #    tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        # Saving checkpoint by checking eval loss was too slow and memory expensive.
                        # if results['eval_loss'] < min_loss:
                        #     min_loss = results['eval_loss']
                        logger.info("Eval loss: {}".format(results['eval_loss']))
                        tb_writer.add_scalar('eval_loss', results['eval_loss'], global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    tb_writer.add_scalar('logging_loss', logging_loss, global_step)
                    tb_writer.add_scalar('loss_item', loss.item(), global_step)
                    logger.info("Loss: {}".format((tr_loss - logging_loss) / args.logging_steps))
                    logger.info("Logging loss: {}".format(logging_loss))
                    logger.info("Loss item: {}".format(loss.item()))
                    logger.info("Global step/Step: {}/{}".format(global_step, step))
                    logging_loss = tr_loss

                    # Save best checkpoint
                    if loss.item() < min_loss:
                        logger.info(
                            "Loss item - Previous: {}, Current min: {}, Global step: {}".format(min_loss, loss.item(),
                                                                                                global_step))
                        min_loss = loss.item()
                        # Save best model checkpoint
                        prefix = 'best'
                        # Find and remove last best
                        list_dirs = os.listdir(args.output_dir)
                        best_dir_ckpt = [s for s in list_dirs if prefix in s]
                        if len(best_dir_ckpt) > 0:
                            shutil.rmtree(os.path.join(args.output_dir, best_dir_ckpt[0]))
                        # Create new checkpoint directory
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}-{}'.format(prefix, global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model,
                                                                'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

                # Save model checkpoint every X steps if save_best = False
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0 and not args.save_best:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix="", train_bertc=False, is_evaluate=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    # eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)
    output_dir = args.output_dir
    if train_bertc:
        output_dir = args.output_dir_complete

    if args.task_name == "mnli":
        eval_outputs_dirs = (output_dir, output_dir + '-MM')
    else:
        if args.eval_output_dir is None:
            eval_outputs_dirs = (output_dir,)
        else:
            eval_outputs_dirs = (args.eval_output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['embracebert', 'bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                          'labels':         batch[3]}

                if args.model_type in ['embracebertwithquery']:
                    outputs = model(**inputs, is_evaluate=is_evaluate)
                else:
                    outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                if args.model_type in ['embracebert', 'embracebertwithquery']:
                    preds = [logits.detach().cpu().numpy()]
                else:  # Why doesn't this work with EmbraceBERT? This is the original line in the code.
                    preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                if args.model_type in ['embracebert', 'embracebertwithquery']:
                    preds.append(logits.detach().cpu().numpy())
                else:
                    preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        results["eval_loss"] = eval_loss

        output_eval_file = os.path.join(eval_output_dir, "{}.txt".format(args.eval_output_filename))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        # Write accuracy in JSON file
        output_eval_file = os.path.join(eval_output_dir, "{}.json".format(args.eval_output_filename))
        with open(output_eval_file, "w") as writer:
            json.dump(results, writer, indent=2)

    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, load_bertc=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    #if args.model_type in ['embracebertwithquert'] and not load_bertc:  # Mixed complete, incomplete data
    #    processor = processors[task](labels_array[args.task_name], load_mixed_data=True)
    #else:
    processor = processors[task](labels_array[args.task_name])
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    data_dir = args.data_dir
    if load_bertc:
        data_dir = args.data_dir_complete

    cached_features_file = os.path.join(data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = processor.get_dev_examples(data_dir) if evaluate else processor.get_train_examples(data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def save_model(args, model, tokenizer, model_class, train_step_type='train', train_bertc=True):
    """ Save additional files in the directory with saved model """
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    output_dir = args.output_dir
    if train_bertc:
        output_dir = args.output_dir_complete

    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(output_dir)

        list_dirs = os.listdir(output_dir)
        if train_step_type == 'train':
            prefix = 'best'
        else:  # pretrain
            prefix = 'pretrain'
        checkpoints = [s for s in list_dirs if prefix in s]
        if len(checkpoints) > 0:
            checkpoint = checkpoints[0]
            output_dir = os.path.join(output_dir, checkpoint)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model,
                                                'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        if args.model_type in ['bert', 'roberta']:  # with args
            model = model_class.from_pretrained(output_dir, dropout_prob=args.dropout_prob)
        elif args.model_type in ['embracebert', 'embraceroberta']:  # with args
            model = model_class.from_pretrained(output_dir, dropout_prob=args.dropout_prob,
                                                is_condensed=args.is_condensed, add_branches=args.add_branches,
                                                share_branch_weights=args.share_branch_weights, p=args.p,
                                                max_seq_length=args.max_seq_length)
        elif args.model_type in ['embracebertwithquery']:  # with args
            model = model_class.from_pretrained(output_dir, dropout_prob=args.dropout_prob,
                                                is_condensed=args.is_condensed, add_branches=args.add_branches,
                                                share_branch_weights=args.share_branch_weights, p=args.p,
                                                max_seq_length=args.max_seq_length,
                                                extract_key_value_from_bertc=args.extract_key_value_from_bertc,
                                                dimension_reduction_method=args.dimension_reduction_method)
        else:
            model = model_class.from_pretrained(output_dir)
        # tokenizer = tokenizer_class.from_pretrained(output_dir)
        model.to(args.device)


def load_model_for_eval(args, model_class , tokenizer_class, train_step_type='train', load_bertc=False):
    output_dir = args.output_dir
    if load_bertc:
        output_dir = args.output_dir_complete

    checkpoints = [output_dir]
    if args.eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    # Find best checkpoint
    list_dirs = os.listdir(output_dir)
    # logger.info("Output dir '{}' has subdirs: {}".format(args.output_dir, list_dirs))
    if train_step_type == 'train':
        prefix = 'best'
    else:
        prefix = train_step_type
    checkpoints = [s for s in list_dirs if prefix in s]

    if len(checkpoints) > 0:
        checkpoint = checkpoints[0]
        output_dir = os.path.join(output_dir, checkpoint)
    # else:
    #    output_dir = os.path.join(args.output_dir, checkpoints[0])

    # for checkpoint in checkpoints:
    global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
    # Load a trained model and vocabulary that you have fine-tuned
    if args.model_type in ['bert', 'roberta']:  # with args
        model = model_class.from_pretrained(output_dir, dropout_prob=args.dropout_prob)
    elif args.model_type in ['embracebert', 'embraceroberta']:  # with args
        model = model_class.from_pretrained(output_dir, dropout_prob=args.dropout_prob, is_condensed=args.is_condensed,
                                            add_branches=args.add_branches,
                                            share_branch_weights=args.share_branch_weights, p=args.p,
                                            max_seq_length=args.max_seq_length)
    elif args.model_type in ['embracebertwithquery']:  # with args
        model = model_class.from_pretrained(output_dir, dropout_prob=args.dropout_prob, is_condensed=args.is_condensed,
                                            add_branches=args.add_branches,
                                            share_branch_weights=args.share_branch_weights, p=args.p,
                                            max_seq_length=args.max_seq_length,
                                            extract_key_value_from_bertc=args.extract_key_value_from_bertc,
                                            dimension_reduction_method=args.dimension_reduction_method)
    else:
        model = model_class.from_pretrained(output_dir)
    tokenizer = tokenizer_class.from_pretrained(output_dir)
    model.to(args.device)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--data_dir_complete", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task - BERTc.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument('--is_condensed', action='store_true',
                        help="In the Embracement Layer, indicates whether to consider all tokens (False)"
                             "or only the ones between tokens CLS and SEP (True)."
                             "Only for EmbraceBERT and EmbraceRoBERTa.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--output_dir_complete", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written - BERTc.")
    parser.add_argument("--log_dir", default=None, type=str,
                        help="The log directory where the model SummaryWriter info will be saved.")
    parser.add_argument("--log_dir_complete", default=None, type=str,
                        help="The log directory where the model SummaryWriter info will be saved.")

    ## Other parameters
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
    parser.add_argument("--eval_type", default="default", type=str,
                        help="Options=[default, incomplete_test]. 'default' refers to when a model is tested with its"
                             " Test Data. 'incomplete_test' refers to when a model is tested with a different test"
                             " data, more specifically, a model that was trained with complete data being tested with"
                             " incomplete data.")
    parser.add_argument("--eval_output_dir", default=None, type=str,
                        help="Only set this when eval_type is 'incomplete_test'")
    parser.add_argument("--eval_output_filename", default="eval_results", type=str)
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--evaluate_steps", default=400, type=int,
                        help="Evaluation steps.")
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
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_best', action='store_true',
                        help="Save best checkpoint.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--apply_dropout', action='store_true',
                        help="Whether to apply dropout after Embrace Layer (only for EmbraceBERT and EmbraceRoBERTa).")
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help="Dropout probability in BERT, RoBERTa, EmbraceBERT/RoBERTa.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    # Parameters to train model with BERT with frozen weights
    parser.add_argument('--freeze_bert_weights', action='store_true',
                        help="Whether to apply freeze BERT weights or not.")
    parser.add_argument("--num_train_epochs_bertc", default=100.0, type=float,
                        help="Total number of training epochs to perform with BERT with frozen weights.")
    parser.add_argument("--num_steps_check_saturated_loss", default=10.0, type=float,
                        help="NOT CURRENTLY IN USE. Total number steps needed to check for saturated loss in order"
                             " to start end-to-end fine-tuning process.")
    parser.add_argument('--train_bertc', action='store_true',
                        help="Whether to add branches in BERT's hidden layers.")
    parser.add_argument('--extract_key_value_from_bertc', action='store_true',
                        help="Whether to use BERTc to extract Q or (K,V).")
    # Add branches
    parser.add_argument('--add_branches', action='store_true',
                        help="Whether to add branches in BERT's hidden layers.")
    parser.add_argument('--share_branch_weights', action='store_true',
                        help="Whether to share weights in branches pooler and classifiers. Classifier's evaluator is "
                             "always shared.")

    # Probability type for EmbraceLayer
    parser.add_argument('--p', type=str, default='multinomial',
                        help="Choose the probability type for p in EmbraceLayer."
                             " Options = ['multinomial': p is random].")

    # Dimension reduction method to consider tokens other than CLS
    parser.add_argument('--dimension_reduction_method', type=str, default='attention',
                        help="Choose the dimension reduction method for EmbraceBERT (CLS token and embrace vector need to become 1 vector)."
                             " Options = ['attention', 'projection'].")

    args = parser.parse_args()

    if os.path.exists(args.output_dir_complete) and os.listdir(args.output_dir_complete) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory for BERTc ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    ensure_dir(args.output_dir_complete)
    ensure_dir(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name](labels_array[args.task_name])
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    model_type_name_has_changed = False
    if args.train_bertc:
        args.model_type = 'bert'
        model_type_name_has_changed = True
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    if args.add_branches == True:  # EmbraceBERT with branches
        config.output_hidden_states = True

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.model_type in ['bert', 'roberta']:  # with args
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config, dropout_prob=args.dropout_prob)
    elif args.model_type in ['embracebert', 'embraceroberta']:  # with args
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config, dropout_prob=args.dropout_prob,
                                            is_condensed=args.is_condensed, add_branches=args.add_branches,
                                            share_branch_weights=args.share_branch_weights, p=args.p,
                                            max_seq_length=args.max_seq_length)
    elif args.model_type in ['embracebertwithquery']:  # with args
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config, dropout_prob=args.dropout_prob,
                                            is_condensed=args.is_condensed, add_branches=args.add_branches,
                                            share_branch_weights=args.share_branch_weights, p=args.p,
                                            max_seq_length=args.max_seq_length,
                                            extract_key_value_from_bertc=args.extract_key_value_from_bertc,
                                            dimension_reduction_method=args.dimension_reduction_method)
    else:
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # ================== Training Full Model =======================
    if model_type_name_has_changed:
        args.model_type = 'embracebertwithquery'
    if args.do_train:
        if args.model_type in ['embracebertwithquery']:
            args.model_type = 'bert'
            train_dataset_complete = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, load_bertc=True)
            # 1. Pre-train BERTc with complete data
            if args.train_bertc:
                global_step_pt1, tr_loss_pt1 = pre_train(args, train_dataset_complete, model, tokenizer, freeze_bert_weights=False, train_bertc=True)
                logger.info(" global_step_pt1 = %s, average loss_pt1 = %s", global_step_pt1, tr_loss_pt1)
                save_model(args, model, tokenizer, model_class, train_step_type='pretrain', train_bertc=True)

            # 2. Load BERTi with BERTc weights and fine-tune on incomplete data (K,V), using Q from BERTc
            model_bertc, tokenizer_bertc = load_model_for_eval(args, model_class, tokenizer_class, train_step_type='pretrain', load_bertc=True)

            args.model_type = 'embracebertwithquery'
            config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
            model_berti, tokenizer_berti = load_model_for_eval(args, model_class, tokenizer_class, train_step_type='pretrain', load_bertc=True)
            train_dataset_incomplete = load_and_cache_examples(args, args.task_name, tokenizer_berti, evaluate=False, load_bertc=False)
            global_step, tr_loss = train(args, train_dataset_complete, train_dataset_incomplete, model_bertc, model_berti, tokenizer_berti)
            logger.info(" global_step_pt2 = %s, average loss_pt2 = %s", global_step, tr_loss)
            save_model(args, model_berti, tokenizer_berti, model_class)
        else:
            train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
            # Pre-train embracement layer with classifier before fine-tuning BERT
            if args.freeze_bert_weights:
                global_step_frozenbert, tr_loss_frozenbert = pre_train(args, train_dataset, model, tokenizer,
                                                                       freeze_bert_weights=True)
                logger.info(" global_step_frozenbert = %s, average loss_frozenbert = %s", global_step_frozenbert,
                            tr_loss_frozenbert)
                save_model(args, model, tokenizer, model_class, train_step_type='pretrain')
                model, tokenizer = load_model_for_eval(args, model_class, tokenizer_class, train_step_type='pretrain')

            global_step, tr_loss = pre_train(args, train_dataset, model, tokenizer, freeze_bert_weights=False)
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            save_model(args, model, tokenizer, model_class)

    # ================== Evaluate Classifier: Test dataset =======================
    results = {}
    # Evaluate full model
    if args.do_eval and args.local_rank in [-1, 0]:
        model, tokenizer = load_model_for_eval(args, model_class, tokenizer_class)
        logger.info("EVAL TYPE: {}".format(args.eval_type))
        if args.eval_type == 'default':
            if args.do_train:
                result = evaluate(args, model, tokenizer, prefix=global_step, is_evaluate=True)
                result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            else:
                result = evaluate(args, model, tokenizer, is_evaluate=True)
                result = dict((k, v) for k, v in result.items())
        else:
            result = evaluate(args, model, tokenizer, is_evaluate=True)
            result = dict((k, v) for k, v in result.items())
        results.update(result)

    # Evaluate BERTc
    if model_type_name_has_changed and args.model_type == 'embracebertwithquery':
        args.model_type = 'bert'
        if args.do_eval and args.local_rank in [-1, 0]:
            model, tokenizer = load_model_for_eval(args, model_class, tokenizer_class)
            logger.info("EVAL TYPE: {}".format(args.eval_type))
            if args.eval_type == 'default':
                if args.do_train:
                    result = evaluate(args, model, tokenizer, prefix=global_step, is_evaluate=True)
                    result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
                else:
                    result = evaluate(args, model, tokenizer, is_evaluate=True)
                    result = dict((k, v) for k, v in result.items())
            else:
                result = evaluate(args, model, tokenizer, is_evaluate=True)
                result = dict((k, v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
