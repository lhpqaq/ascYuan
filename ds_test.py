import logging
import os
import json
import argparse
import torch
import torch.nn as nn
import deepspeed
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from deepspeed.runtime.lr_schedules import WarmupLR
from model.transformer2 import Transformer
#from model.Transformer_model import Transformer
from tqdm import tqdm
from megatron import mpu
from megatron.fp16 import FP16_Module
#import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from DataSet import TensorDataset
crossentropyloss=nn.CrossEntropyLoss(reduction='sum')

class Yuan_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cel=nn.CrossEntropyLoss(reduction='sum')
    def forward(self, x, y):
        batch_size = x.size(0)
        loss = 0
        for logit,target in zip(x,y):
            loss += self.cel(logit,target)
        return loss

class My_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nll=torch.nn.NLLLoss(reduction='mean')
    def forward(self, x, y):
        batch_size = x.size(0)
        x = torch.nn.functional.softmax(x,dim=1)
        x = torch.log(x)
        loss = 0
        for logit,target in zip(x,y):
            loss += self.nll(logit,target)
        return loss
myloss = My_loss()

yuan_loss = Yuan_loss()

def create_moe_param_groups(model):
    from deepspeed.moe.utils import is_moe_param

    params_with_weight_decay = {'params': [], 'name': 'weight_decay_params'}
    moe_params_with_weight_decay = {
        'params': [],
        'moe': True,
        'name': 'weight_decay_moe_params'
    }

    for module_ in model.modules():
        moe_params_with_weight_decay['params'].extend([
            p for n, p in list(module_._parameters.items())
            if p is not None and is_moe_param(p)
        ])
        params_with_weight_decay['params'].extend([
            p for n, p in list(module_._parameters.items())
            if p is not None and not is_moe_param(p)
        ])

    return params_with_weight_decay, moe_params_with_weight_decay

def get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-b',
                        '--batch_size',
                        default=32,
                        type=int,
                        help='mini-batch size (default: 32)')
    parser.add_argument('-e',
                        '--epochs',
                        default=30,
                        type=int,
                        help='number of total epochs (default: 30)')
    parser.add_argument("--local_rank",
                        type=int, 
                        default=0,
                        help='local rank passed from distributed launcher')
    parser.add_argument('--log-interval',
                        type=int,
                        default=2000,
                        help="output logging information at a given interval")
    parser.add_argument('--moe',
                        default=False,
                        action='store_true',
                        help='use deepspeed mixture of experts (moe)')
    parser.add_argument('--ep-world-size',
                        default=1,
                        type=int,
                        help='(moe) expert parallel world size')
    parser.add_argument('--num-experts',
                        default=1,
                        type=int,
                        help='(moe) number of total experts')
    parser.add_argument('--top-k',
                        default=1,
                        type=int,
                        help='(moe) gating top 1 and 2 supported')
    parser.add_argument('--zero', 
                        type=int, 
                        default=0)
    parser.add_argument('--moe-param-groups',
                        default=False,
                        action='store_true',
                        help='(moe) create separate moe param groups, required when using ZeRO w. MoE') 
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help='use fp16')
    parser.add_argument('--model_parallel_size', 
                        type=int, 
                        default=1,
                        help='model parallel size')
    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args() 

    return args

def main():
    logging.basicConfig(level=logging.INFO #设置日志输出格式
                    ,filename="demo2.log" #log日志输出的文件位置和文件名
                    ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" #日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
                    )

    args = get_args()

    deepspeed.init_distributed(verbose=False)
    mpu.initialize_model_parallel(args.model_parallel_size)
    if args.moe:
        deepspeed.utils.groups.initialize(ep_size=args.ep_world_size, mpu=mpu)

    model = Transformer(seq_len=2048,vocab_size=53228,N=40,d_ff=3072,h=24,d_model=480)

    #To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    if args.fp16:
        model = FP16_Module(model)

    #i = torch.cuda.current_device()
    #model = DDP(model, device_ids=[i], output_device=i,
    #            process_group=mpu.get_data_parallel_group())

    #parameters = get_params_for_weight_decay_optimization(model)


    if args.moe_param_groups:
        parameters = create_moe_param_groups(model)

    lr_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = WarmupLR(lr_optimizer)

    model_engine, optimizer, _, lr_scheduler= deepspeed.initialize(args=args,
                                                                    model=model,
                                                                    model_parameters=model.parameters(),
                                                                    lr_scheduler=lr_scheduler,
                                                                    mpu=mpu,
                                                                    dist_init_required=True,
                                                                    config=args.deepspeed_config)

    print('Checkpoint loading...')
    model_engine.load_checkpoint('./checkpoint/')
    print('Checkpoint loading completed!')
    #data_loader = get_data_loader(path='data/test_index.txt',batch_size=args.batch_size)
    print('Training data loading...')
    torch_data=TensorDataset('./data/iData0.npy',if_dir=False)
    #torch_data=TensorDataset('./data/',if_dir=True)
    datas = DataLoader(torch_data, batch_size=args.batch_size, shuffle=False, drop_last=False)
    print('Training data loading completed!')
    #torch.distributed.broadcast(data_loader,
    #                            mpu.get_model_parallel_src_rank(),
    #                            group=mpu.get_model_parallel_group())
    model_engine.save_checkpoint('./checkpoint/')
    loop = tqdm(enumerate(datas),total=len(datas))
    for i,Input in loop:
        loop.set_description('Train')
        if args.fp16:
            Input = Input.half()
        Input = Input.to(model_engine.device).long()
        #Input = torch.LongTensor(Input)
        out=model_engine(Input)
        #out = out.view(-1,out.shape[-1])
        #Input = Input.view(-1).long()
        #print('Input:',Input.shape)
        #print('out:',out.shape)
        #loss = crossentropyloss(out,Input)
        loss = myloss(out,Input)
        log = 'loss:' + str(loss.item()/args.batch_size) + '\t\ttokens_num:' + str((i+1)*args.batch_size*2048/100000) + 'HT'
        logging.info(log)
        loop.set_postfix(loss=loss.item()/args.batch_size)
        #print('loss:\n',loss.item()*2048)
        model_engine.backward(loss)
        model_engine.step()
        if(i%1000==0):
            model_engine.save_checkpoint('./checkpoint/')
        #lr_scheduler.step()

if __name__ == "__main__":
    main()
