import torch
import torch.nn as nn
import torch
from self_attention_pool import SelfAttentiveSpanExtractor
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        #self.dense1 = nn.Linear(100+100, config.hidden_size)
        #self.dense1 = nn.Linear(config.hidden_size, 100)
        self.dense2 = nn.Linear(config.hidden_size, 150)
        #self.dense3 = nn.Linear(config.hidden_size, 100)
        self.dense4 = nn.Linear(config.hidden_size, 150)
        #self.dense5 = nn.Linear(config.hidden_size, 100)
        self.dense6 = nn.Linear(300, 300)
        self.dense7 = nn.Linear(600, 300)
        #self.dense3 = nn.Linear(100, 100)
        #self.dense2 = nn.Linear(config.hidden_size*2, config.hidden_size)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(300, 2)
        # self.normlayer1 = nn.LayerNorm(200)
        # self.normlayer2 = nn.LayerNorm(config.hidden_size)
        # self.normlayer3 = nn.LayerNorm(config.hidden_size)
        # self.normlayer4 = nn.LayerNorm(config.hidden_size)
        # self.normlayer5 = nn.LayerNorm(300)
        # self.normlayer6 = nn.LayerNorm(config.hidden_size)
        # self.normlayer7 = nn.LayerNorm(config.hidden_size)
       # self.normlayer2 = nn.LayerNorm(config.hidden_size*2)
        #self.normlayer3 = nn.LayerNorm(config.hidden_size*2)
       # self.normlayer4 = nn.LayerNorm(config.hidden_size)
     
    def forward(self, span_diff_l1, emb_diff_l1, mutant_op ):
       
        
        #span_diff_l1 = self.normlayer2(span_diff_l1)
        span_diff_l1 = self.dense2(span_diff_l1)
        
        #span_diff_l2 = self.normlayer3(span_diff_l2)
        #emb_2 = self.dense3(emb_2)
        
        #emb_diff_l1 = self.normlayer6(emb_diff_l1)
        emb_diff_l1 = self.dense4(emb_diff_l1)
        
        #emb_diff_l2 = self.normlayer7(emb_diff_l2)
       # emb_diff_l2 = self.dense5(emb_diff_l2)
        

        mutant_op = self.dense6(mutant_op)
     
        x = torch.cat( ( mutant_op, span_diff_l1, emb_diff_l1), 1 )
        #x = self.normlayer5(x)
        
        x = self.dropout(x)
        x = self.dense7(x)
        #x = torch.cat((x, mutant_op), 1)
        #x = self.normlayer1(x)
        #x = self.normlayer1(x)
        x = torch.relu(x)
        x = self.dropout(x)
       # x=self.normlayer3(x)
       # x = self.dropout(x)
        # x = self.dense2(x)
        # x = torch.relu(x)
        # x=self.normlayer4(x)
       # x = self.dropout(x)
        x = self.out_proj(x)
        return x




class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.selfattetionpool=SelfAttentiveSpanExtractor(config.hidden_size)
        self.mutant_mebedding = nn.Embedding(11, 300)
        self.args=args
        self.normlayer = nn.LayerNorm(config.hidden_size)
        self.normlayer1 = nn.LayerNorm(config.hidden_size)
    
        
    def forward(self, inputs_ids_1,attn_mask_1,span_1,inputs_ids_2,attn_mask_2,span_2,mtype, labels=None): 
       
        # pass the inputs to the model
        #https://huggingface.co/docs/transformers/main/en/main_classes/output
        #print(inputs_ids_1.shape)
        hidden1, emb_1 = self.encoder(inputs_ids_1, attention_mask=attn_mask_1)[:2]
        #emb_1 = self.normlayer(emb_1)
        #print(emb_1.shape)
        #print(span_1.shape)
        hideen2, emb_2 = self.encoder(inputs_ids_2, attention_mask=attn_mask_2)[:2]
        #emb_2 = self.normlayer(emb_2)
        emb_span_1 = self.selfattetionpool(hidden1, span_1).squeeze(1)
        emb_span_2 = self.selfattetionpool(hideen2, span_2).squeeze(1)
        #emb_span_1 = self.normlayer1(emb_span_1)
        #emb_span_2 = self.normlayer1(emb_span_2)
        #print(emb_2.shape)
        #outputs1 = torch.cat( (emb_1, emb_span_1), 1 )
        #outputs2 = torch.cat( (emb_2, emb_span_2), 1 )
        span_diff_l1 = emb_span_1 - emb_span_2
        #span_diff_l2 = emb_span_1 - emb_span_2

        emb_diff_l1 = emb_1 - emb_2
        #emb_diff_l2 = emb_1 - emb_2

        
        
        mutant_op=self.mutant_mebedding(mtype)
       # outputs2 = torch.cat( (emb_2, mutant_op), 1 )
     #   print(outputs.shape)
        logits=self.classifier(span_diff_l1, emb_diff_l1, mutant_op)
        # shape: [batch_size, num_classes]
        prob=F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob

class RobertaClassificationHeadToken(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size) 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)
      #  self.normlayer1 = nn.LayerNorm(config.hidden_size*2)
      
     
    def forward(self, x ):
       # x=self.normlayer1(x)
        x = self.dropout(x)
        x = self.dense1(x)
        x = torch.relu(x)
   
        x = self.out_proj(x)
        return x

class ModelAnnoted(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(ModelAnnoted, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHeadToken(config)
        self.hidden_size = config.hidden_size
        self.args=args
    
        
    def forward(self, inputs_ids, attn_mask, mask_token_index, labels=None):    
        # pass the inputs to the model
        #https://huggingface.co/docs/transformers/main/en/main_classes/output
        emb = self.encoder(inputs_ids, attention_mask=attn_mask)[0]
        bs,nt, fs=emb.size()
        #import torch
        #x = torch.arange(24).view(4, 3, 2) 
        #print(x)
        #ids = torch.randint(0, 3, size=(4, 1))
        #print(ids.shape)
        #idx = ids.repeat(1, 2).view(4, 1, 2) 
        #print(idx.shape)
        #a=torch.gather(x, 1, idx)
        #print("gather size")
        #print(a.shape)
        #print(idx.size())
        #fids=ids.flatten()
        #print(fids)
        #print(x[0, fids[0], :])
        #print(x[1, fids[1], :])
        #print(x[2, fids[2], :])
        #print(x[3, fids[3], :])
        index = mask_token_index.repeat(1, fs).view(bs, 1, fs)
        #print(emb.shape)
        #print(index.shape)
        #print(mask_token_index.max())
        outputs  = torch.gather(emb, 1, index).squeeze(1)
        logits=self.classifier(outputs)
        # shape: [batch_size, num_classes]
        prob=F.softmax(logits, dim=-1)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob
        else:
            return prob
