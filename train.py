from torch.utils.data import DataLoader
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

from model import My_model
from dataset import FruitDataset


if __name__=='__main__':
    max_epoch = 500
    cur_iter = 0

    model = My_model()

    train_dataset = FruitDataset('./Datasets/train', is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=16,
                              num_workers=8, shuffle=True, persistent_workers=True, drop_last=True, pin_memory=True)

    test_dataset = FruitDataset('./Datasets/test', is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1,
                              num_workers=1, shuffle=True, persistent_workers=True, drop_last=True, pin_memory=True)

    model = model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": 1e-3},
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]}]
    
    opt = torch.optim.AdamW(param_dicts, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max_epoch, eta_min=5e-7, last_epoch=-1)
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()
    os.makedirs('checkpoints', exist_ok=True)

    best_loss = 1000
    for cur_epoch in range(max_epoch):
        model.train()
        for sample in train_loader:
            image = sample['image'].cuda()
            heatmap = sample['heatmap'].cuda()
            heatmap_size = sample['heatmap_size'].cuda()

            opt.zero_grad()

            pred_heat, pred_heat_size = model(image)
            mse_loss = mse(pred_heat.squeeze(1), heatmap)
            mse_size_loss = mse(pred_heat_size.squeeze(1), heatmap_size)
            loss = mse_loss * 0.8 + mse_size_loss

            loss.backward()
            opt.step()
            cur_iter += 1
            if cur_iter % 100 == 0:
                print(f'epoch {cur_epoch}/{max_epoch} iter {cur_iter} total loss {loss.item()} mse loss {mse_loss.item()}  size loss {mse_size_loss.item()}')

        if cur_epoch % 10 == 0:
            torch.save({'model': model.state_dict()}, os.path.join('checkpoints', f'epoch_{cur_epoch}.pth'))
        scheduler.step()

        model.eval()
        vis_idx = random.randint(0, len(test_loader) - 1)
        losses = []
        with torch.no_grad():
            for idx, sample in enumerate(test_loader):
                image = sample['image'].cuda()
                heatmap = sample['heatmap'].cuda()
                heatmap_size = sample['heatmap_size'].cuda()
                pred_heat, pred_heat_size = model(image)
                mse_loss = mse(pred_heat.squeeze(1), heatmap)
                mse_size_loss = mse(pred_heat_size.squeeze(1), heatmap_size)
                loss = (mse_loss + mse_size_loss).item()
                losses.append(loss)
                if idx == vis_idx:
                    image = image.cpu().numpy()[0].transpose((1, 2, 0)) * 0.5 + 0.5
                    left_1 = pred_heat.cpu().numpy()[0][0]
                    left_2 = pred_heat_size.cpu().numpy()[0][0]
                    plt.subplot(131)
                    plt.imshow(image)
                    plt.subplot(132)
                    plt.imshow(left_1)
                    plt.subplot(133)
                    plt.imshow(left_2)
                    os.makedirs('vis', exist_ok=True)
                    plt.savefig(os.path.join('vis', f'{cur_iter}.png'))
                    plt.close()

        # if best_loss > np.mean(losses):
        #     best_loss = np.mean(losses)
        #     print(f"model saved, best loss {best_loss} at epoch {cur_epoch}")
        #     torch.save({'model': model.state_dict()}, os.path.join('checkpoints', f'best.pth'))
