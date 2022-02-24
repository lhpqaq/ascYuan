import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, embed, pad_size, dropout, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pe = torch.tensor(
            [[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out



class Transformer(nn.Module):
	def __init__(self,
         		 seq_len = 2048,
				 vocab_size = 53228,
				 N = 40,
				 d_model = 480,
				 d_ff = 3072,
				 h = 24,
				 d_k = 64,
				 d_v = 64,
				 dropout = 0.1,
				 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
				 ):
		super(Transformer,self).__init__()
		self.embedding = nn.Embedding(vocab_size,d_model)
		self.position_embedding = PositionalEncoding(d_model,seq_len,dropout,device)
		self.encoder_layer = nn.TransformerEncoderLayer(d_model,h,d_ff,dropout)
		self.encoder = nn.TransformerEncoder(self.encoder_layer,N)
		self.wv = nn.Linear(d_model,vocab_size, bias = False)

	def forward(self,x):
		Input = self.embedding(x[0])
		Input = self.position_embedding(Input).unsqueeze(0)
		for i in range(1,len(x)):
			temp = self.embedding(x[i])
			temp = self.position_embedding(temp).unsqueeze(0)
			Input = torch.cat((Input,temp),0)
		out = self.encoder(Input)
		Output = self.wv(out)
		return Output
