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
from models import HashingTwoBranch

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
    p.add_argument("--center_loss_weight", type=float, default=1e-2)

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
    batch_x_png = torch.stack([data[1] for data in batch])
    batch_y = torch.tensor([data[2] for data in batch]).squeeze()
    batch_x_raw_len = torch.tensor([len(x) for x in batch_x_raw]).long()
    batch_x_raw = rnn_utils.pad_sequence(batch_x_raw, batch_first=True)
    return (batch_x_raw, batch_x_png, batch_y, batch_x_raw_len)


if __name__ == '__main__':
    args = parse_args()
    if args.device == 'gpu':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    seed_everything(args.seed)


    model = HashingTwoBranch(3, (1, 28, 28), 25).to(device)
    clf_loss_func = nn.CrossEntropyLoss()
    center_loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if not args.predict_only:
        train_dataset = TwoBranchDataset('train', args.data_path_raw, args.data_path_png)
        valid_dataset = TwoBranchDataset('valid', args.data_path_raw, args.data_path_png)
        train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, args.batch_size, collate_fn=collate_fn)
        total_train_steps = len(train_loader)
        total_valid_steps = len(valid_loader)
        best_acc = 0.
        print('[HashingTwoBranch] Train begin!')

        print('[HashingTwoBranch] Step 1: Pretrain the classification network.')
        for ep in range(1, args.epoch + 1):
            model.train()
            for i, batch in enumerate(train_loader):
                batch_x_raw, batch_x_png, batch_y, batch_x_raw_len = batch
                batch_x_raw = batch_x_raw.to(device)
                batch_x_png = batch_x_png.to(device)
                batch_y = batch_y.to(device)
                batch_x_raw_len = batch_x_raw_len

                code, logits = model(batch_x_raw, batch_x_png, batch_x_raw_len)
                loss = clf_loss_func(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'\r[HashingTwoBranch][Epoch {ep}/{args.epoch}] Training > {i + 1}/{total_train_steps} Loss: {loss.item():.3f}', end='')
            print()
            
            with torch.no_grad():
                y_true, y_pred = [], []
                model.eval()
                for i, batch in enumerate(valid_loader):
                    batch_x_raw, batch_x_png, batch_y, batch_x_raw_len = batch
                    batch_x_raw = batch_x_raw.to(device)
                    batch_x_png = batch_x_png.to(device)
                    batch_y = batch_y
                    batch_x_raw_len = batch_x_raw_len

                    code, logits = model(batch_x_raw, batch_x_png, batch_x_raw_len)
                    pred = logits.argmax(1)
                    y_true.append(batch_y.data.numpy())
                    y_pred.append(pred.data.cpu().numpy())
                    print(f'\r[HashingTwoBranch][Epoch {ep}/{args.epoch}] Validating > {i + 1}/{total_valid_steps} ...', end='')
                y_true = np.concatenate(y_true)
                y_pred = np.concatenate(y_pred)
                acc = accuracy_score(y_true, y_pred)
                print('done. Validation accuracy: %.4f' % acc)

            if acc > best_acc:
                print(f'[HashingTwoBranch][Epoch {ep}/{args.epoch}] *** New best! *** Accuracy: {acc:.4f}')
                path = join(args.save_path, 'HashingTwoBranch.bin')
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model.state_dict(), path)
                best_acc = acc

        print('[HashingTwoBranch] Step 2: Calculate class feature centers.')
        unnoisy_dataset = TwoBranchDataset('train', args.data_path_raw, args.data_path_png, remove_noise=True)
        unnoisy_loader = DataLoader(unnoisy_dataset, args.batch_size, collate_fn=collate_fn)
        total_unnoisy_steps = len(unnoisy_loader)
        class_centers = torch.zeros(25, 250)
        class_count = torch.tensor([np.sum(unnoisy_dataset.y == i) for i in range(25)]).long().unsqueeze(-1)
        path = join(args.save_path, 'HashingTwoBranch.bin')
        with torch.no_grad():
            model.load_state_dict(torch.load(path))
            model.eval()
            for i, batch in enumerate(unnoisy_loader):
                batch_x_raw, batch_x_png, batch_y, batch_x_raw_len = batch
                batch_x_raw = batch_x_raw.to(device)
                batch_x_png = batch_x_png.to(device)
                batch_y = batch_y
                batch_x_raw_len = batch_x_raw_len

                code, logits = model(batch_x_raw, batch_x_png, batch_x_raw_len)
                class_centers[batch_y] += code.data.cpu()
                print(f'\r[HashingTwoBranch] Calculating center > {i + 1}/{total_unnoisy_steps} ...', end='')
            class_centers /= class_count
            print('done.')

        print('[HashingTwoBranch] Step 3: Finetune the model using the centers.')
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr / 10)
        class_centers = class_centers.to(device)
        for ep in range(1, args.epoch + 1):
            model.train()
            for i, batch in enumerate(train_loader):
                batch_x_raw, batch_x_png, batch_y, batch_x_raw_len = batch
                batch_x_raw = batch_x_raw.to(device)
                batch_x_png = batch_x_png.to(device)
                batch_y = batch_y.to(device)
                batch_x_raw_len = batch_x_raw_len

                code, logits = model(batch_x_raw, batch_x_png, batch_x_raw_len)
                loss_clf = clf_loss_func(logits, batch_y)
                loss_center = center_loss_func(code, class_centers[batch_y])
                loss = loss_clf + args.center_loss_weight * loss_center
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'\r[HashingTwoBranch][Epoch {ep}/{args.epoch}] Training > {i + 1}/{total_train_steps} Loss: {loss.item():.3f}', end='')
            print()
            
            with torch.no_grad():
                y_true, y_pred = [], []
                model.eval()
                for i, batch in enumerate(valid_loader):
                    batch_x_raw, batch_x_png, batch_y, batch_x_raw_len = batch
                    batch_x_raw = batch_x_raw.to(device)
                    batch_x_png = batch_x_png.to(device)
                    batch_y = batch_y
                    batch_x_raw_len = batch_x_raw_len

                    code, logits = model(batch_x_raw, batch_x_png, batch_x_raw_len)
                    pred = logits.argmax(1)
                    y_true.append(batch_y.data.numpy())
                    y_pred.append(pred.data.cpu().numpy())
                    print(f'\r[HashingTwoBranch][Epoch {ep}/{args.epoch}] Validating > {i + 1}/{total_valid_steps} ...', end='')
                y_true = np.concatenate(y_true)
                y_pred = np.concatenate(y_pred)
                acc = accuracy_score(y_true, y_pred)
                print('done. Validation accuracy: %.4f' % acc)

            if acc > best_acc:
                print(f'[HashingTwoBranch][Epoch {ep}/{args.epoch}] *** New best! *** Accuracy: {acc:.4f}')
                path = join(args.save_path, 'HashingTwoBranch.bin')
                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model.state_dict(), path)
                best_acc = acc


    print('[HashingTwoBranch] Test begin!')
    test_dataset = TwoBranchDataset('test', args.data_path_raw, args.data_path_png)
    test_loader = DataLoader(test_dataset, args.batch_size, collate_fn=collate_fn)
    total_test_steps = len(test_loader)
    path = join(args.save_path, 'HashingTwoBranch.bin')
    with torch.no_grad():
        model.load_state_dict(torch.load(path))
        model.eval()
        y_true, y_pred = [], []
        for i, batch in enumerate(test_loader):
            batch_x_raw, batch_x_png, batch_y, batch_x_raw_len = batch
            batch_x_raw = batch_x_raw.to(device)
            batch_x_png = batch_x_png.to(device)
            batch_y = batch_y
            batch_x_raw_len = batch_x_raw_len

            code, logits = model(batch_x_raw, batch_x_png, batch_x_raw_len)
            pred = logits.argmax(1)
            y_true.append(batch_y.data.numpy())
            y_pred.append(pred.data.cpu().numpy())
            print(f'\r[HashingTwoBranch] Testing > {i + 1}/{total_test_steps} ...', end='')
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
    plt.savefig('figures/HashingTwoBranch.png')
