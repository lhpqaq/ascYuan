from megatron.tokenizer import tokenization_enc_dec
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torch
import os
import numpy as np
#from transformers import AutoTokenizer
seq_length=2048
vocab = []
fr = open("vocab.txt",encoding="utf-8")
for line in fr:
    vocab.append(line.strip())
fr.close()
vocab_size = len(vocab)

word2index = { w: i for i,w in enumerate(vocab) }
index2woed = { i: w for i,w in enumerate(vocab) }

#tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese") 
tokenizer = tokenization_enc_dec.EncDecTokenizer(vocab_file="vocab.txt")

class TensorDataset(Dataset):

    def __init__(self, data_path,if_dir=True):
        if(if_dir): #路径是个文件夹
            files= os.listdir(data_path)
            data_tensor = []
            for file in files:
                data = np.load(data_path+'/'+file, allow_pickle=True)
                for a in data:
                    a1 = np.pad(a,(0,2048-len(a)),'constant', constant_values=(0,0))
                    data_tensor.append(a1.astype(int))
            data_tensor = np.array(data_tensor)
            data_tensor = torch.tensor(data_tensor)      
        else:      #路径是一个文件      
            data = np.load(data_path, allow_pickle=True)
            data_tensor = []
            for a in data:
                a1 = np.pad(a,(0,2048-len(a)),'constant', constant_values=(0,0))
                data_tensor.append(a1.astype(int))
            data_tensor = np.array(data_tensor)
            data_tensor = torch.tensor(data_tensor)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # data_tensor = data_tensor.to(device)
        self.data_tensor = data_tensor.long()

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)






