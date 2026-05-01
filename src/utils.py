
import os
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Config:
    def __init__(self):
        self.seed = 42
        self.device = get_device()
        self.epochs = 200
        self.lr = 0.001
        self.weight_decay = 5e-4
        self.dropout = 0.5
        self.hidden_dim = 64
        self.output_dim = 5 # Number of classes (BRCA has 5 subtypes usually: Basal, Her2, LumA, LumB, Normal)
        
        # Data paths
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'BRCA')
        self.prior_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Prior Knowledge')
        
        self.omics_files = ['1_tr.csv', '2_tr.csv', '3_tr.csv']
        self.label_file = 'labels_tr.csv'
