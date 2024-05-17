
import argparse
import torch
import os
import tqdm

import numpy as np

from dataset import RefineDataset
from model import RefineNet

def inference(args):
    print(args)

    device = "cpu" if args.gpu<0 else f"cuda:{args.gpu}"

    dataset = RefineDataset(seq_len=args.seq_len, stride=args.stride, mode=args.mode)
    
    model = RefineNet(
        num_layers=args.nlayer, 
        latent_dim=args.latent_dim,
        feedforward_dim=args.feedforward_dim,
        nhead=args.nhead,
        dropout=args.dropout
    ).to(device)

    weights = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(weights['model'])

    experiment_name = args.checkpoint.split('/')[-2]
    save_path = os.path.join(args.save_path, experiment_name)

    os.makedirs(save_path, exist_ok=True)

    model = model.eval()

    if args.seed >= 0:
        np.random.seed(args.seed)

    if args.num_samples == -1:
        idx = np.arange(len(dataset))
    else:
        idx = np.random.permutation(len(dataset))[:args.num_samples]

    iterator = tqdm.tqdm(idx)
    with torch.no_grad():
        for i in iterator:
            data = dataset[i]

            iterator.set_description(f"{data['filename']}, {data['input'].shape}")

            x = data['input'].unsqueeze(0).to(device) # 1, seq_len, dim
            y = data['output'].unsqueeze(0).to(device) # 1, seq_len, 3

            pred = model(x) # 1, seq_len, 3

            pred = torch.cat([pred, x], -1).squeeze(0).detach().cpu().numpy() # 1, seq_len, 3+dim 
            true = torch.cat([y, x], -1).squeeze(0).detach().cpu().numpy() # 1, seq_len, 3+dim

            np.save(os.path.join(save_path, f"{data['filename']}_pred"), pred)
            np.save(os.path.join(save_path, f"{data['filename']}_true"), true)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=-1, help="-1:CPU")
    parser.add_argument("--save_path", type=str, default="./infer", help="")

    # 데이터셋 파라미터
    parser.add_argument("--seq_len", type=int, default=120)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--mode", type=int, default=2)

    # 샘플링 파라미터
    parser.add_argument("--seed", type=int, default=42, help="-1:랜덤")
    parser.add_argument("--num_samples", type=int, default=4, help="-1:전체 데이터셋 샘플링")

    # 모델 파라미터
    parser.add_argument("--nlayer", type=int, default=8, help="트랜스포머 레이어 수")
    parser.add_argument("--latent_dim", type=int, default=512, help="")
    parser.add_argument("--feedforward_dim", type=int, default=1024, help="")
    parser.add_argument("--nhead", type=int, default=8, help="")
    parser.add_argument("--dropout", type=float, default=0.1, help="")
    parser.add_argument("--checkpoint", type=str, default="./experiments/0/best.pt", help="")
    args = parser.parse_args()

    inference(args)

