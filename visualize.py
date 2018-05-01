import torch
import torch.nn.functional as F
import os
from model import DeepJ
from tqdm import tqdm
import dataset
import constants as const
import argparse

def main():
    parser = argparse.ArgumentParser(description='Visualize model latent space.')
    parser.add_argument('model', help='Path to model file')
    args = parser.parse_args()

    data, labels = dataset.process(dataset.load())
    labels = labels.cpu().numpy().tolist()

    model = DeepJ().cuda()
    model.load_state_dict(torch.load(args.model))

    with open(os.path.join(const.OUT_DIR, 'latent.tsv'), 'w') as f:
        for d in tqdm(data):
            d = d.unsqueeze(0).cuda()
            d = model.embd(d)
            mean, _, _ = model.encoder(d, None)
            mean = mean.squeeze(0).data.cpu().numpy().tolist()
            f.write('\t'.join(map(str, mean)) + '\n')

    with open(os.path.join(const.OUT_DIR, 'labels.tsv'), 'w') as f:
        f.write('\n'.join(map(str, labels)))

if __name__ == '__main__':
    main()
