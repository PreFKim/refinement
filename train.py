import random
import torch
import argparse
import tqdm
import os
import glob

import numpy as np 
import torch.nn as nn
from torch.utils.data import DataLoader

from model import RefineNet
from dataset import RefineDataset

def set_seed(seed):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def steps(epoch, dataloader, model, optimizer, criterion, device, mode=0):

    model = model.to(device)

    desc = f"Epoch {epoch}|{'Train' if mode==0 else 'Valid'}"
    iterator = tqdm.tqdm(iterable=dataloader, desc=desc)

    summation_loss = 0.0

    for i, data in enumerate(iterator):
        x = data['input'].to(device)
        y = data['output'].to(device)

        output = model(x)

        loss = criterion(output, y)

        summation_loss = summation_loss + loss.detach().cpu().numpy()
        if mode == 0:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        iterator.set_description(f"{desc}|MSE:{summation_loss/(i+1)}")

    return summation_loss



def train(args):

    print(args)

    save_dir = os.path.join(args.save_dir, str(len(glob.glob(os.path.join(args.save_dir,"*")))))

    train_set = RefineDataset(
        args.seq_len,
        args.stride,
        mode=0,
    )
    valid_set = RefineDataset(
        args.seq_len,
        args.stride,
        mode=1,
    )

    print(f"Num Train: {len(train_set)}, Num Valid: {len(valid_set)}")

    device = "cpu" if args.gpu == -1 else f"cuda:{args.gpu}"

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    model = RefineNet(
        num_layers=args.nlayer, 
        latent_dim=args.latent_dim,
        feedforward_dim=args.feedforward_dim,
        nhead=args.nhead,
        dropout=args.dropout
    )

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.learning_rate)

    criterion = nn.MSELoss()

    if args.checkpoint != "" :
        ckpt = torch.load(args.checkpoint, map_location=device)

        model.load_state_dict(ckpt['model'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer'], strict=True)

    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    valid_losses = []
    for i in range(args.epochs):
        model = model.train()
        train_losses.append(steps(i+1, train_loader, model, optimizer, criterion, device, mode=0))
        model = model.eval()
        with torch.no_grad():
            valid_losses.append(steps(i+1, valid_loader, model, optimizer, criterion, device, mode=1))


        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        torch.save(ckpt, os.path.join(save_dir, "last.pt"))
        
        if valid_losses[-1] <= min(valid_losses):
            torch.save(ckpt, os.path.join(save_dir, "best.pt"))
            print(f"Best Validation Loss {i+1} : {valid_losses[i]}")
            print(f"Saved at {os.path.join(save_dir, 'best.pt')}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--save_dir", type=str, default="./experiments", help="")

    # 데이터셋 파라미터
    parser.add_argument("--seq_len", type=int, default=120, help="데이터 시퀀스 길이")
    parser.add_argument("--stride", type=int, default=10, help="데이터 프레임 스트라이드")

    # 모델 파라미터
    parser.add_argument("--nlayer", type=int, default=8, help="트랜스포머 레이어 수")
    parser.add_argument("--latent_dim", type=int, default=512, help="")
    parser.add_argument("--feedforward_dim", type=int, default=1024, help="")
    parser.add_argument("--nhead", type=int, default=8, help="")
    parser.add_argument("--dropout", type=float, default=0.1, help="")
    parser.add_argument("--checkpoint", type=str, default="", help="")

    # 학습 파라미터
    parser.add_argument("--random_seed", type=int, default=42, help="fix random seed")
    parser.add_argument("--num_workers", type=int, default=2, help="")
    parser.add_argument("--epochs", type=int, default=1000, help="")
    parser.add_argument("--batch_size", type=int, default=64, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="")
    parser.add_argument("--gpu", type=int, default=1, help="")

    args = parser.parse_args()

    set_seed(args.random_seed)
    train(args)
    
    
    
