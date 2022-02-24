import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
from torch.autograd import Variable

class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model  
 
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
class Positional_Encoding(nn.Module): 
  "Implement the PE function." 
  def __init__(self, d_model,max_len,dropout,device): 
    #d_model=512,dropout=0.1,
    #max_len=5000代表事先准备好长度为5000的序列的位置编码，其实没必要，
    #一般100或者200足够了。
    super(Positional_Encoding, self).__init__() 
    self.dropout = nn.Dropout(p=dropout) 

    # Compute the positional encodings once in log space. 
    pe = torch.zeros(max_len, d_model) 
    #(5000,512)矩阵，保持每个位置的位置编码，一共5000个位置，
    #每个位置用一个512维度向量来表示其位置编码
    position = torch.arange(0, max_len).unsqueeze(1) 
    # (5000) -> (5000,1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * 
      -(math.log(10000.0) / d_model)) 
      # (0,2,…, 4998)一共准备2500个值，供sin, cos调用
    pe[:, 0::2] = torch.sin(position * div_term) # 偶数下标的位置
    pe[:, 1::2] = torch.cos(position * div_term) # 奇数下标的位置
    pe = pe.unsqueeze(0) 
    # (5000, 512) -> (1, 5000, 512) 为batch.size留出位置
    self.register_buffer('pe', pe) 
  def forward(self, x): 
    x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False) 
    # 接受1.Embeddings的词嵌入结果x，
    #然后把自己的位置编码pe，封装成torch的Variable(不需要梯度)，加上去。
    #例如，假设x是(30,10,512)的一个tensor，
    #30是batch.size, 10是该batch的序列长度, 512是每个词的词嵌入向量；
    #则该行代码的第二项是(1, min(10, 5000), 512)=(1,10,512)，
    #在具体相加的时候，会扩展(1,10,512)为(30,10,512)，
    #保证一个batch中的30个序列，都使用（叠加）一样的位置编码。
    return self.dropout(x) # 增加一次dropout操作
# 注意，位置编码不会更新，是写死的，所以这个class里面没有可训练的参数。


# class Positional_Encoding(nn.Module):
#     def __init__(self, embed, pad_size, dropout, device):
#         super(Positional_Encoding, self).__init__()
#         self.device = device
#         self.pe = torch.tensor(
#             [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
#         self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
#         self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
#         out = self.dropout(out)
#         return out

class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  
        out = self.layer_norm(out)
        return out

class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''

    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        '''
        Args:
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale:
        Return:
        '''
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        # if mask:  # TODO change this
        #     attention = attention.masked_fill_(mask == 0, -1e9)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context

class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        # if mask:  # TODO
        #     mask = mask.repeat(self.num_head, 1, 1)  # TODO change this
        scale = K.size(-1) ** -0.5  
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        #print('x--',x.shape)
        out = self.fc(context)
        #out = out.view(x.shape[0],x.shape[1])
        #print('4--',out.shape)
        out = self.dropout(out)
        #print('5--',out.shape)
        out = out + x  
        out = self.layer_norm(out)
        return out

class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        #print('2--',out.shape)
        out = self.feed_forward(out)
        #print('3--',out.shape)        
        return out

class Transformer(nn.Module):
    def __init__(self,
                 seq_len=100,
                 vocab_size = 1000,
                 N = 2,
                 d_model = 300,
                 d_ff = 512,
                 h = 5,
                 dropout = 0.1,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 ):
        super(Transformer,self).__init__()
        self.embedding = Embeddings(vocab_size,d_model)
        self.position_embedding = Positional_Encoding(d_model,seq_len,dropout,device)
        self.encoder = Encoder(d_model,h,d_ff,dropout)
        self.encoders = nn.ModuleList([copy.deepcopy(self.encoder) for _ in range(N)])
        self.wv = nn.Linear(d_model,vocab_size, bias = False)

    # def forward(self,x):
    #     out = self.embedding(x)
    #     out = self.position_embedding(out)
    #     for encoder in self.encoders:
    #         out = encoder(out)
    #     out = self.wv(out)
    #     return out
    def forward(self,x):
        out = self.embedding(x)
        out = self.position_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = self.wv(out)
        #out = out.view(x.size(0),-1,2048)
        return out