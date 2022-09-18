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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
from __future__ import absolute_import, division, print_function

import argparse
from genericpath import isfile
import glob
from lib2to3.pgen2 import token
import logging
from opcode import opname
import os
import pickle
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torchsampler import ImbalancedDatasetSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)
from tqdm import tqdm, trange
from custom_tokenizer import custom_special_tokens_dict, mutant_operators, abstraction_vocab
from models import ModelAnnoted
from transformers import AutoTokenizer, AutoModel, AutoConfig

# To control logging level for various modules used in the application:
import logging
import re
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
set_global_logging_level(logging.ERROR, ["transformers"])  
cpu_cont = 16
logger = logging.getLogger(__name__)         
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
             input_tokens,
             input_ids,
             source_mask,
             annotaion_start_token_index,
             annonation_end_token_index,
             mask_token_index,
             project,
             global_id,
             label
    ):
        #The first code function
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.source_mask = source_mask
        self.annotaion_start_token_index = annotaion_start_token_index 
        self.annonation_end_token_index = annonation_end_token_index
        self.mask_token_index = mask_token_index
      
        
        #label
        self.label=label
        self.global_id = global_id
        self.project = project
       


def convert_examples_to_features(item):
    #source
    
    code1, mask_index, annotaion_start, annonation_end, label,tokenizer, args, project, global_id=item
    
    #print(code1)
   
    tokenized_code = tokenizer(code1)   
    mask_token_index = tokenized_code.char_to_token(mask_index)-1
    annotaion_start_token_index = tokenized_code.char_to_token(annotaion_start)-1
    #print([annotaion_start, annonation_end])
    #print(code1[annotaion_start:annonation_end])
    b=tokenized_code.char_to_token(annonation_end-1)
    assert b!= None, f"={code1[annonation_end-1]}= \n {[annotaion_start, annonation_end]} \n {code1[annotaion_start:annonation_end]} \n {code1}"
    annonation_end_token_index = tokenized_code.char_to_token(annonation_end-1)-1
    max_token_len = args.code_length-2
    

    assert mask_token_index is not None and annotaion_start_token_index is not None and annonation_end_token_index is not None, f'{mask_index},{mask_token_index}, \n {tokenizer.convert_tokens_to_string(tokenized_code.tokens())} \n {open("debug.c", "w").write(code1)}'
    source_tokens = tokenized_code.tokens()[1:-1]
    c = 0
    if annonation_end_token_index > max_token_len-1:
        l = annonation_end_token_index - annotaion_start_token_index
        m = mask_token_index - annotaion_start_token_index
        source_tokens = source_tokens[ max(0, annotaion_start_token_index-100) : min(annonation_end_token_index+100, len(source_tokens)) ]
        annotaion_start_token_index = min(annotaion_start_token_index, 100)
        annonation_end_token_index = annotaion_start_token_index + l
        mask_token_index = annotaion_start_token_index + m
        c = 1
    #assert mask_token_index < 512, f"{open('tmp.txt', 'w').write(code1)}"
    #assert annonation_end_token_index < 512, f"{open('tmp.txt', 'w').write(code1)}"
    #assert annotaion_start_token_index < 512, f"{open('tmp.txt', 'w').write(code1)}"
    if mask_token_index >= 512 or annonation_end_token_index >= 512 or annotaion_start_token_index >= 512:
        with open('tmp.txt', 'a') as f:
                s=[project, global_id, code1]
                ss=json.dumps(s)
                f.write(f"{ss}\n")
        return [[],2]
    source_tokens = source_tokens[:args.code_length-2]
    source_tokens =[tokenizer.cls_token]+source_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens) 
    source_mask = [1] * (len(source_tokens))
    padding_length = args.code_length - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    source_mask+=[0]*padding_length     
    assert source_tokens[mask_token_index+1] == "<mask>", f"{source_tokens[mask_token_index+1]} {len(source_tokens[mask_token_index+1])}"
    ex = InputFeatures(source_tokens,source_ids,source_mask,annotaion_start_token_index,annonation_end_token_index,mask_token_index,
                         project,global_id, label )

    return [ex, c]

from multiprocessing import Pool
from tqdm import *

def imap_unordered_bar(func, args, n_processes = 20):
    p = Pool(processes=None)
    res_list = []
    counter_truncted = 0
    dropped = 0
    with tqdm(total = len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            if res[1] == 2:
                dropped += 1
                continue
            counter_truncted += res[1]
            pbar.update()
            pbar.set_description(f"truncted {counter_truncted}/{i+1}, dropepd {dropped}")
            res_list.append(res[0])
            
    pbar.close()
    p.close()
    p.join()
    return res_list

import collections

def create_data( tokenizer, args, dataname, file_path='train', testproject=[]):
    datapath = os.path.join('data', f'{dataname}_bert_cache.pt')
    stat_path =  os.path.join('data', f'{dataname}_test_{"_".join(testproject)}_stat.json')
  
    #load indedx
    index_filename=file_path
    logger.info("Creating features from index file at %s ", index_filename)
    data = []
    if not os.path.isfile(datapath):
        with open(file_path, "r") as f:
            for line in f:
                line=line.strip()
                js=json.loads(line)
                print(js)
                print(list(js.keys()))
                project_name = js['meta']['project']
                #code1, mask_index, annotaion_start, annonation_end, label,tokenizer, args, project, global_id
                data.append( (js['operation_code'], js['mask_location'], js["annotaion_start"], js["annotaion_end"],
                                 js['label'], tokenizer, args, project_name, js['meta']['global_id'] ) )   
        #print(data[:2])    
        random.shuffle(data)
        examples= imap_unordered_bar(convert_examples_to_features, data[:89250])
        examples = [e for e in examples if e is not None ]
        pickle.dump(examples, open(datapath, "wb"))
    else:
        examples= pickle.load( open(datapath, "rb"))
    if "bert" in args.dataname:
        shuffle_index_path =  os.path.join('data', f'{dataname}_shuffle.json')
        if not os.path.isfile(shuffle_index_path):
            indexes = [ i for i in range(len(examples)) ]
            random.shuffle(indexes)
            json.dump(indexes, open(shuffle_index_path, "w") )
        else:
            indexes = json.load(open(shuffle_index_path, "r"))
        indexes = indexes[:int(0.2*len(indexes))]
        examples = [ examples[j] for j in indexes ]

    if debug :
        examples=examples[:1000]

    if args.sampling:
        examples=examples[:89250]
    test_examples = []
    train_examples = []
    check_list = []
    valid_examples = []
    projects_list = []
  #  data_stat={1:0, 0:0}
    for e in tqdm(examples, total=len(examples)):
        if e.project in testproject:
            test_examples.append(e)
       #     logger.info(f"data stat {data_stat, data_stat[1]/(data_stat[0]+data_stat[1])}")
            if e.project not in check_list:
                check_list.append( e.project )
        else:
            if e.project not in projects_list:
                projects_list.append( e.project )
   # assert len(check_list) == 3, f"Wrong test projects {check_list}, {testproject}"
    random.shuffle(projects_list)
    for e in tqdm(examples, total=len(examples)):
        # if e.project in projects_list[:3]:
        #     valid_examples.append(e)
        if e.project in projects_list:
            train_examples.append(e)
    random.shuffle(train_examples)
    tl=[ e.label for e in train_examples]
    testlabels = [ e.label for e in test_examples ]
    tdata_stat=collections.Counter(tl)
    test_stat = collections.Counter(testlabels)
    json.dump({"train":tdata_stat,"train_ratio":tdata_stat[1]/(tdata_stat[0]+tdata_stat[1]),  "test":test_stat, 
        "test_ratio": test_stat[1]/(test_stat[0]+test_stat[1]), "testp":check_list }, open(stat_path, "w"), indent=4)
    logger.info(f"train data stat {tdata_stat, tdata_stat[1]/(tdata_stat[0]+tdata_stat[1])}")
    logger.info(f"test data stat {test_stat, test_stat[1]/(test_stat[0]+test_stat[1])}")
    valid_examples = train_examples[int(len(train_examples)*0.8):]
    train_examples = train_examples[:int(len(train_examples)*0.8)]
    return (train_examples, valid_examples, test_examples)

class TextDataset(Dataset):
    def __init__(self, args, examples,tokenizer, isPrint=False):
        self.examples = examples
        self.args=args
        if isPrint:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))  
                logger.info("mask index: {}".format(example.mask_token_index+1))
                logger.info("mask ids: {}".format( example.input_ids[example.mask_token_index+1]))
                logger.info("subsuming ids: {}".format( example.input_ids[example.mask_token_index]))
                logger.info("mask token: {}".format([x.replace('\u0120','_') for x in example.input_tokens[example.mask_token_index+1]]))
                #logger.info("mask id: {}".format(' '.join(map(str, example.input_ids[example.mask_token_index])))) 

      
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):   
        #self.annotaion_start_token_index = annotaion_start_token_index 
        #self.annonation_end_token_index = annonation_end_token_index
        #self.mask_token_index = mask_token_index       
        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(self.examples[item].source_mask), 
                torch.tensor([self.examples[item].annotaion_start_token_index]),
                torch.tensor([self.examples[item].annonation_end_token_index]),   
                torch.tensor([self.examples[item].mask_token_index+1]),       
                torch.tensor(self.examples[item].label))
    
    def getlabels(self):
         return  [ e.label for e in self.examples ]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset,valid_examples, test_examples, model, tokenizer):
    """ Train the model """
    
    #build dataloader
    train_sampler = ImbalancedDatasetSampler(train_dataset, labels=train_dataset.getlabels())
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=0)
    
    args.max_steps=args.epochs*len( train_dataloader)
    args.save_steps=len( train_dataloader)//10
    args.warmup_steps=args.max_steps//5
    model.to(args.device)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step=0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_f1=0
    best_mcc=0
    model.zero_grad()
    middle_res = {}
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            (inputs_ids_1,attn_mask_1, at,
            ae,mi,
            labels)=[x.to(args.device)  for x in batch]
            model.train()
            loss,logits = model(inputs_ids_1,attn_mask_1,mi,labels)

            if args.n_gpu > 1:
                loss = loss.mean()
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
                
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer,valid_examples, eval_when_training=True)  
                    logger.info("***** Eval results *****")
                    for key in sorted(results.keys()):
                        logger.info("  %s = %s", key, str(round(results[key],4)))  
                    #test_examples
                    results_test = evaluate(args, model, tokenizer,test_examples, eval_when_training=True)  
                    middle_res[f"{len(middle_res)}"] = {"val": results, "test":results_test}
                    json.dump(middle_res, open( os.path.join(args.output_dir, "middle_res.json"), "w"))
                    logger.info("***** Test results *****")
                    for key in sorted(results_test.keys()):
                        logger.info("  %s = %s", key, str(round(results_test[key],4))) 
                    # Save model checkpoint
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                    
                    # Save model checkpoint
                    if results['eval_mcc']>best_mcc:
                        best_mcc=results['eval_mcc']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best MCC:%s",round(best_mcc,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-mcc'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)
                        
def evaluate(args, model, tokenizer,examples, eval_when_training=False, best_threshold=0.5):
    #build dataloader
    eval_dataset = TextDataset(args, examples, tokenizer)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,batch_size=args.eval_batch_size,num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:    
        (inputs_ids_1,attn_mask_1, at,
            ae,mi,
            labels)=[x.to(args.device)  for x in batch]
  
        with torch.no_grad():
            lm_loss,logit = model(inputs_ids_1,attn_mask_1,mi,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    #calculate scores
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    #best_threshold=0.5
    best_f1=0

    y_preds=logits[:,1]>best_threshold
    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds)   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds)  
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_trues, y_preds) 
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_trues, y_preds)           
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_mcc": float(mcc),
        "eval_acc":float(acc),
        "eval_threshold":best_threshold,
        
    }

    

    return result

def test(args, model, tokenizer,examples, best_threshold=0):
    #build dataloader
    eval_dataset = TextDataset( args,examples, tokenizer)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in eval_dataloader:
        (inputs_ids_1,attn_mask_1, at,
            ae,mi,
            labels)=[x.to(args.device)  for x in batch]
  
        with torch.no_grad():
            lm_loss,logit = model(inputs_ids_1,attn_mask_1,mi,labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    
    #output result
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    y_preds=logits[:,1]>best_threshold

    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds)
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds)   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds)  
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(y_trues, y_preds)    
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_trues, y_preds)       
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_mcc": float(mcc),
        "eval_acc":float(acc),
        "eval_threshold":best_threshold,
        
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))
    json.dump(result, open( os.path.join(args.output_dir,"predictions_performance_mcc.json"), "w" ) , indent=4)
    with open(os.path.join(args.output_dir,"predictions.txt"),'w') as f:
        for example,pred in zip(eval_dataset.examples,y_preds):
            if pred:
                f.write(example.global_id+'\t'+'1'+'\n')
            else:
                f.write(example.global_id+'\t'+'0'+'\n')
                                                
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    #parser.add_argument("--eval_data_file", default=None, type=str,
    #                    help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    #parser.add_argument("--test_data_file", default=None, type=str,
    #                    help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_abstraction", action='store_true',
                        help="add abstraction token to the tokenizer")

    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--sampling", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_freeze", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
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
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--nl_length", default=256, type=int,
                        help="Optional NL input sequence length after tokenization.")    
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")
    parser.add_argument('--test_projects', nargs='+', help='Test Projects', required=True)
    parser.add_argument("--dataname", default=None, type=str, required=True,
                        help="bert or ibir")
   

    args = parser.parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu,)


    # Set seed
    set_seed(args)
    config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, return_dict=False, output_hidden_states=True)
    tokenizer =  AutoTokenizer.from_pretrained(args.tokenizer_name)
    # add the speical tokens
    num_added = tokenizer.add_special_tokens(custom_special_tokens_dict)
    num_added_token = 0
    if args.do_abstraction:
        assert len(abstraction_vocab) > 0
        logger.info(f"abstraction tokens {len(abstraction_vocab)}")
        num_added_token = tokenizer.add_tokens(abstraction_vocab+mutant_operators)
    logger.info(f"Added num of special tokens {num_added}  Added common tokens {num_added_token}")
    # create dataset
    logger.info("Load Dataset")
    (train_examples, valid_examples, test_examples) = create_data( tokenizer, args, args.dataname, file_path=args.data_file, testproject=args.test_projects)

    logger.info("Finishing loading Dataset")
    encoder = AutoModel.from_pretrained(args.model_name_or_path,config=config) 
    logger.info(f"Resize embedding layer {num_added}")
    encoder.resize_token_embeddings(len(tokenizer))
    model=ModelAnnoted(encoder,config,tokenizer,args)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = TextDataset( args, train_examples, tokenizer, isPrint=True)
        train(args, train_dataset,valid_examples, test_examples,model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result=evaluate(args, model, tokenizer, valid_examples)
        
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        test(args, model, tokenizer,test_examples,best_threshold=0.5)

    return results


if __name__ == "__main__":
    debug=False
    main()
