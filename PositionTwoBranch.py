import argparse
import os
import random
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from torch.utils.data import DataLoader

from data_processor import TwoBranchDataset
from models import PositionTwoBranch

def parse_args():
    parser = argparse.ArgumentParser()

    p = parser.add_argument_group("General")
    p.add_argument("--device", type=str, default='cpu')
    p.add_argument("--data_path_raw", type=str, default='./datasets/quickdraw')
    p.add_argument("--data_path_png", type=str, default='./datasets/png')

    p = parser.add_argument_group("Model")
    p.add_argument("--load_path", type=str, default='./saved_models')
    p.add_argument("--save_path", type=str, default='./saved_models')

    p = parser.add_argument_group("Train")
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epoch", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)

    p = parser.add_argument_group("Predict")
    p.add_argument("--predict_only", default=False, action='store_true')

    return parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    batch_x_raw = [data[0] for data in batch]
    batch_x_abs = [data[1] for data in batch]
    batch_x_png = torch.stack([data[2] for data in batch])
    batch_y = torch.tensor([data[3] for data in batch]).squeeze()
    batch_x_raw_len = torch.tensor([len(x) for x in batch_x_raw]).long()
    batch_x_raw = rnn_utils.pad_sequence(batch_x_raw, batch_first=True)
    batch_x_abs = rnn_utils.pad_sequence(batch_x_abs, batch_first=True)
    return (batch_x_raw, batch_x_abs, batch_x_png, batch_y, batch_x_raw_len)


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    seed_everything(args.seed)

    train_dataset = TwoBranchDataset('train', args.data_path_raw, args.data_path_png, to_abs=True)
    valid_dataset = TwoBranchDataset('valid', args.data_path_raw, args.data_path_png, to_abs=True)
    test_dataset = TwoBranchDataset('test', args.data_path_raw, args.data_path_png, to_abs=True)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, args.batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, args.batch_size, collate_fn=collate_fn)

    model = PositionTwoBranch(3, (1, 28, 28), 25).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not args.predict_only:
        total_train_steps = len(train_loader)
        total_valid_steps = len(valid_loader)
        best_acc = 0.
        print('[PositionTwoBranch] Train begin!')
        for ep in range(1, args.epoch + 1):
            model.train()
            for i, batch in enumerate(train_loader):
                batch_x_raw, batch_x_abs, batch_x_png, batch_y, batch_x_raw_len = batch
                batch_x_raw = batch_x_raw.to(device)
                batch_x_abs = batch_x_abs.to(device)
                batch_x_png = batch_x_png.to(device)
                batch_y = batch_y.to(device)
                batch_x_raw_len = batch_x_raw_len

                logits = model(batch_x_raw, batch_x_abs, batch_x_png, batch_x_raw_len)
                loss = loss_func(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'\r[PositionTwoBranch][Epoch {ep}/{args.epoch}] Training > {i + 1}/{total_train_steps} Loss: {loss.item():.3f}', end='')
            print()
            
            with torch.no_grad():
                y_true, y_pred = [], []
                model.eval()
                for i, batch in enumerate(valid_loader):
                    batch_x_raw, batch_x_abs, batch_x_png, batch_y, batch_x_raw_len = batch
                    batch_x_raw = batch_x_raw.to(device)
                    batch_x_abs = batch_x_abs.to(device)
                    batch_x_png = batch_x_png.to(device)
                    batch_y = batch_y
                    batch_x_raw_len = batch_x_raw_len

                    logits = model(batch_x_raw, batch_x_abs, batch_x_png, batch_x_raw_len)
                    pred = logits.argmax(1)
                    y_true.append(batch_y.data.numpy())
                    y_pred.append(pred.data.cpu().numpy())
                    print(f'\r[PositionTwoBranch][Epoch {ep}/{args.epoch}] Validating > {i + 1}/{total_valid_steps} ...', end='')
                y_true = np.concatenate(y_true)
                y_pred = np.concatenate(y_pred)
                acc = accuracy_score(y_true, y_pred)
                print('done. Validation accuracy: %.4f' % acc)

            if acc > best_acc:
                print(f'[PositionTwoBranch][Epoch {ep}/{args.epoch}] *** New best! *** Accuracy: {acc:.4f}')
                path = join(args.save_path, 'PositionTwoBranch.bin')
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model.state_dict(), path)
                best_acc = acc

    print('[PositionTwoBranch] Test begin!')
    total_test_steps = len(test_loader)
    path = join(args.save_path, 'PositionTwoBranch.bin')
    with torch.no_grad():
        model.load_state_dict(torch.load(path))
        model.eval()
        y_true, y_pred = [], []
        for i, batch in enumerate(test_loader):
            batch_x_raw, batch_x_abs, batch_x_png, batch_y, batch_x_raw_len = batch
            batch_x_raw = batch_x_raw.to(device)
            batch_x_abs = batch_x_abs.to(device)
            batch_x_png = batch_x_png.to(device)
            batch_y = batch_y
            batch_x_raw_len = batch_x_raw_len

            logits = model(batch_x_raw, batch_x_abs, batch_x_png, batch_x_raw_len)
            pred = logits.argmax(1)
            y_true.append(batch_y.data.numpy())
            y_pred.append(pred.data.cpu().numpy())
            print(f'\r[PositionTwoBranch] Testing > {i + 1}/{total_test_steps} ...', end='')
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        acc = accuracy_score(y_true, y_pred)
        print('done. Test accuracy: %.4f' % acc)
    plt.rcParams['figure.figsize'] = (16, 12)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, 
        y_pred, 
        normalize=None, 
        display_labels=test_dataset.label_name, 
        xticks_rotation=45,
        # values_format='.3f'
    )
    # plt.show()
    plt.savefig('figures/PositionTwoBranch.png')
