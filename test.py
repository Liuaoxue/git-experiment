import dgl
import numpy as np
import torch
from torch import nn
import torch as th
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import random
import tqdm
import sklearn.metrics
from torch import cosine_similarity
from dataset import *
from model import *
from utils import *
#Hyper-parameters

d_node=128
epoch=args.epochs
K=args.multihead     #multi_head
lambda_1=args.lambda_1
lambda_2=args.lambda_2
lambda_3=args.lambda_3

if __name__ == '__main__':
    g,friend_list_index_test=data(d_node, 'NYC')
    g = g.to(device)
    user_emb=torch.tensor(np.load("data/save_user_embedding/best_auc_JK0.5088859196776176.npy")).to(device)
    test_auc, ap, top_k = test(user_emb, g, friend_list_index_test)
    print("test_auc:",test_auc)
    print("test_ap:",ap)