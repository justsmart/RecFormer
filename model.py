import torch
import torch.nn as nn 
# from Layers import EncoderLayer, DecoderLayer
# from Embed import Embedder, PositionalEncoder

import copy
import math
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F



def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    
def setEmbedingModel(d_list,d_out):
    
    return nn.ModuleList([nn.Linear(d,d_out) for d in d_list])
def setReEmbedingModel(d_list,d_out):
    return nn.ModuleList([nn.Linear(d_out,d)for d in d_list])
class Mlp(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.2):
        super(Mlp, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)

        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout1:
            out = self.dropout2(out)
        return out

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):

    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k) #scores shape is [bs heads view view]
    if mask is not None:
        mask = mask.unsqueeze(1).float()
        mask = mask.unsqueeze(-1).matmul(mask.unsqueeze(-2))#mask shape is [bs 1 view view]
        # mask = mask.unsqueeze(1) #mask shape is [bs 1 1 view]
        scores = scores.masked_fill(mask == 0, -1e9) # mask invalid view
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):

        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model/h
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.2):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.dropout_1(F.relu(self.linear_1(x)))
        x = self.dropout_2(self.linear_2(x))
        return x
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)

        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, \
        src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        print(x)
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Parameter(self.pe[:,:seq_len])
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)
class Encoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N

        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, src, mask):
        x = src
        # x = self.embed(src)
        # x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N

        # self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder( d_model, N, heads, dropout)
        # self.decoder = Encoder(src_vocab, d_model, 1, heads, dropout)
        # self.out = nn.Linear(d_model, trg_vocab) 
    def forward(self, src, src_mask):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")

        # d_output = self.decoder(e_outputs, src_mask)
        # output = self.out(d_output)
        return e_outputs

class TransformerWoDecoder(nn.Module):
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder( d_model, N, heads, dropout)
        self.decoder = Decoder( d_model, N, heads, dropout)
        # self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, src, src_mask=None):
        e_outputs = self.encoder(src, src_mask)
        #print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        # output = self.out(e_outputs)
        return d_output
class Model(nn.Module):
    def __init__(self, input_len, d_model, n_layers, heads, d_list, classes_num, dropout,recover=True):
        super().__init__()
        self.ETrans = Transformer(d_model, n_layers, heads, dropout)
        self.DTrans = Transformer(d_model, n_layers, heads, dropout)
        self.embeddinglayers = setEmbedingModel(d_list,d_model)
        self.re_embeddinglayers = setReEmbedingModel(d_list,d_model)
        self.recover = recover
        # self.classifier = nn.Linear(d_model,classes_num)
    def forward(self,x,mask=None):
        view_num = len(x)
        for i in range(view_num): # encode input view to features with same dimension 
            x[i] = self.embeddinglayers[i](x[i])
        
        # x=self.embeddinglayers[0](x[0]).unsqueeze(1)
        x = torch.stack(x,dim=1) # B,view,d
        x = self.ETrans(x,mask) if self.recover else self.ETrans(x,None)
        encX = x
        # H = torch.einsum('bvd->bd',H)
        if self.recover: # Stage 1
            x = x.mul(mask.unsqueeze(2))
            x = torch.einsum('bvd->bd',x)
            wei = 1 / torch.sum(mask, 1)
            x = torch.diag(wei).mm(x)
        else: # Stage 2
            x = torch.einsum('bvd->bd',x)/view_num
        H = x.detach().clone()
        # ori_x = x.detach().clone()
        x = x.unsqueeze(1).expand(-1,view_num,-1)
        x = self.DTrans(x,None)
        decX = x
        # H[(1-mask).bool()] = x[(1-mask).bool()].clone().detach() 

        x_bar = [None]*view_num
        for i in range(view_num):       
            x_bar[i] = self.re_embeddinglayers[i](x[:,i])
        
        # x_bar = torch.stack(x_bar,dim=1) # B,view,d
        # print(x_bar[0][0,:].detach().cpu().numpy().shape)
        return encX,decX,x_bar,H,None,None


def get_model( d_list,d_model=768,n_layers=2,heads=4,classes_num=10,dropout=0.2,load_weights=None,device=torch.device('cuda:0')):
    
    assert d_model % heads == 0
    assert dropout < 1

    # model = Transformer(input_len, output_len, d_model, n_layers, heads, dropout)
    model = Model(len(d_list), d_model, n_layers, heads, d_list, classes_num, dropout)

    if load_weights is not None:
        print("loading pretrained weights...")
        # model.load_state_dict(torch.load(f'{opt.load_weights}/model_weights'))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    
    model = model.to(device)
    
    return model



