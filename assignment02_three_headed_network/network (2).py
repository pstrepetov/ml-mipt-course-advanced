import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

import tqdm

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Reorder(nn.Module):
    def forward(self, input):
        return input.permute((0, 2, 1))

class ThreeInputsNet(nn.Module):
    def __init__(self, n_tokens, n_cat_features, concat_number_of_features, hid_size=64):
        super(ThreeInputsNet, self).__init__()
        self.title_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.title_encoder = nn.Sequential(
          nn.Conv1d(in_channels=hid_size, out_channels=hid_size, kernel_size=2),
          nn.ReLU(),
          nn.AdaptiveAvgPool1d(output_size=1),
          Flatten()
        )
        
        self.full_emb = nn.Embedding(n_tokens, embedding_dim=hid_size)
        self.full_encoder = nn.Sequential(
          nn.Conv1d(in_channels=hid_size, out_channels=hid_size, kernel_size=2),
          nn.ReLU(),
          nn.AdaptiveAvgPool1d(output_size=1),
          Flatten()
        )
        
        self.category_out = nn.Sequential(
          nn.Linear(n_cat_features, 600),
          nn.ReLU(),
          nn.Linear(600, 100),
          nn.ReLU(),
          nn.Linear(100, 20)
        )


        # Example for the final layers (after the concatenation)
        self.inter_dense = nn.Linear(in_features=concat_number_of_features, out_features=hid_size*2)
        self.final_dense = nn.Linear(in_features=hid_size*2, out_features=1)

        

    def forward(self, whole_input):
        input1, input2, input3 = whole_input
        title_beg = self.title_emb(input1).permute((0, 2, 1))
        title = self.title_encoder(title_beg)
        
        full_beg = self.full_emb(input2).permute((0, 2, 1))
        full = self.full_encoder(full_beg)     
        
        category = self.category_out(input3)      
        
        concatenated = torch.cat(
            [
            title.view(title.size(0), -1),
            full.view(full.size(0), -1),
            category.view(category.size(0), -1)
            ],
            dim=1)
        
        out = self.inter_dense(concatenated)
        out = nn.ReLU()(out)
        out = self.final_dense(out)
        
        return out