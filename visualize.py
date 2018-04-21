import torch
import torch.nn as nn
import torch.nn.functional as F
from model import EncoderRNN, DecoderRNN, DeepJ
from tqdm import tqdm
from dataset import *

data, labels = process(load())
labels = labels.cpu().numpy().tolist()

model = DeepJ().cuda()
model.load_state_dict(torch.load('out/model_VAE.pt'))

with open('latent.tsv', 'w') as f:
    for d in tqdm(data):
        d = var(d.unsqueeze(0), volatile=True)
        d = model.embd(d)
        mean, logvar, _ = model.encoder(d, None)
        mean = mean.squeeze(0).data.cpu().numpy().tolist()
        f.write('\t'.join(map(str, mean)) + '\n')

with open('labels.tsv', 'w') as f:
    f.write('\n'.join(map(str, labels)))
