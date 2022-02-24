import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from megatron.tokenizer import tokenization_enc_dec
#from transformers import AutoTokenizer


seq_len = 2048
vocab = []
fr = open("vocab.txt",encoding="utf-8")
for line in fr:
    vocab.append(line.strip())
fr.close()
vocab_size = len(vocab)

word2index = { w: i for i,w in enumerate(vocab) }
index2word = { i: w for i,w in enumerate(vocab) }

#tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese") 
tokenizer = tokenization_enc_dec.EncDecTokenizer(vocab_file="vocab.txt")

train_data = []
fr = open("data/test0.txt",encoding="utf-8")
for line in fr:
    temp = line.strip()
    cut_list = tokenizer.tokenize(temp)
    #cut = jieba.cut(temp)
    #cut_list = [x for x in cut]
    if len(cut_list) == 0:
        continue
    if len(cut_list) > seq_len:
        cut_list = cut_list[:seq_len]
    word_index = []
    for i,word in enumerate(cut_list):
        index = 0
        if word in vocab:
            index = word2index[word]
            word_index.append(index)
    train_data.append(torch.tensor(word_index))
fr.close()
train_data.append(torch.zeros(seq_len))
train_data = pad_sequence([train_data[i] for i in range(len(train_data))],batch_first=True)

fr = open("data/test0_index.txt",'a')
for i in range(len(train_data)-1):
    s = train_data[i].tolist()
    s = str(s).replace('[','').replace(']','')+'\n'
    fr.write(s)
fr.close()
