

import torch

from torch.utils.data import DataLoader

import os

import numpy as np

import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment



from model import My_model

from dataset import FruitDataset



def euclidean_distance(A, B):

    return np.linalg.norm(A[:, np.newaxis] - B, axis=2)



def best_match(A, B):

    m, n = len(A), len(B)

    cost_matrix = euclidean_distance(A, B)

    if m > n:

        cost_matrix = np.pad(cost_matrix, ((0, 0), (0, m - n)), mode="constant", constant_values=1e10)

    elif n > m:

        cost_matrix = np.pad(cost_matrix, ((0, n - m), (0, 0)), mode="constant", constant_values=1e10)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    return row_ind, col_ind





def find_local_peak(heat, kernel=3):

    hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=kernel // 2)

    keep = (hmax == heat).float()

    return heat * keep



class EntropyLoss(torch.nn.Module):

    def __init__(self):

        super(EntropyLoss, self).__init__()



    def forward(self, heatmap):

        # 计算热力图的信息熵 Calculate entropy of heat map

        N, C, H, W = heatmap.size()

        entropy = torch.zeros(N).to(device=heatmap.device)

        for n in range(N):

            point_count = 0

            for c in range(C):

                temp_heatmap = heatmap[n, c]

                temp_heatmap = temp_heatmap[temp_heatmap>0]

                point_count += len(temp_heatmap)

                probs = torch.nn.Softmax(dim=1)(temp_heatmap.view(-1,len(temp_heatmap)))

                entropy[n] += (-probs * torch.log2(probs)).sum()

            entropy[n] /= point_count



        # 返回信息熵作为损失 return the entropy as the loss

        return entropy



if __name__ == '__main__':



    # 模型路径和名称 Model path and name

    model_path = './checkpoints'

    model_name = 'epoch_380.pth'

    # 找出loss最大或最小的 topk 比例样本   Find Top k minimum/maximum; Max or Min entropy

    topk = 0.2

    # 选择最大或最小 find max or min 

    find_max = False

    # 重复更新 affine p 次     Affine transformation times

    p = 2

    # 按照训练时的 batch_size 16 做TTA 

    traing_batch_size = False



    model = My_model()

    test_dataset = FruitDataset('./Datasets/test', is_train=False)

    test_loader = DataLoader(test_dataset, batch_size=1,

                             num_workers=2, shuffle=False, persistent_workers=True, drop_last=True, pin_memory=True)

    

    weights_dict = torch.load(os.path.join(model_path, model_name))

    model.load_state_dict(weights_dict['model'])

    model = model.cuda()



    getLoss = EntropyLoss()



    if traing_batch_size:

        # 按照训练时的batch_size 过网络  Pass the network according to the batch_size during training

        test_loader1 = DataLoader(test_dataset, batch_size=16,

                            num_workers=2, shuffle=False, persistent_workers=True, drop_last=True, pin_memory=True)



        with torch.no_grad():

            model.train()

            for idx, sample in enumerate(test_loader1):

                image = sample['image'].cuda()

                pred_heat, pred_heat_size = model(image)

    else:

        # 第一遍算信息熵，找出信息熵小的样本 Calculate entropy in the first round and find samples with small entropy

        model.eval()

        loss = None

        with torch.no_grad():

            for idx, sample in enumerate(test_loader):

                image = sample['image'].cuda()

                pred_heat, pred_heat_size = model(image)



                if loss is None:

                    # loss = getLoss(pred_heat) + getLoss(pred_heat_size)

                    loss = getLoss(pred_heat)

                else:

                    # loss = torch.cat((loss, getLoss(pred_heat) + getLoss(pred_heat_size)), dim=0)

                    loss = torch.cat((loss, getLoss(pred_heat)), dim=0)

            # print(loss)



        #找出loss最小的k%的样本的序号  Find the ID number of the k% sample with the smallest loss

        loss = loss.detach().cpu().numpy()

        index = np.argsort(loss)

        if find_max:

            index = index[::-1]

        index = index[:int(len(index)*topk)]

        # print(index)



        # 第二遍使用信息熵小的样本进行TTA  The second round uses samples with small entropy for TTA.

        model.train()

        with torch.no_grad():

            image = None

            for idx, sample in enumerate(test_loader):

                if idx in index:

                    if image is None:

                        image = sample['image'].cuda()

                    else:

                        image = torch.cat((image, sample['image'].cuda()), dim=0)



            #重复更新 affine p 次  Affine transformation times

            for i in range(p):

                pred_heat, pred_heat_size = model(image)



    # 第三遍再过网络，得出最终的结果 Get the final results from the thrid round.

    model.eval()

    with torch.no_grad():

        cur_iter = 0

        ct = 0

        diff = 0

        miss = 0

        add = 0

        diff_size = 0

        for idx, sample in enumerate(test_loader):

            image = sample['image'].cuda()

            heatmap = sample['heatmap'].cuda()

            heatmap_size = sample['heatmap_size'].cuda()

            pred_heat, pred_heat_size = model(image)



            # fig = plt.figure(figsize=(19.2, 10.8))



            left_1 = pred_heat.cpu().numpy()[0][0]

            # plt.subplot(143)

            # plt.title('heatmap output')

            # plt.imshow(left_1)



            rst = find_local_peak(pred_heat)

            rst = rst.cpu().numpy()[0][0]

            rst[rst <= 0.4] = 0

            rst[rst > 0] = 255

            rst = rst.astype(np.uint8)

            location = np.where(rst > 0)

            # plt.subplot(144)

            # plt.title('peak')

            # plt.imshow(rst)



            # plt.subplot(142)

            # plt.title('heatmap gt')

            # plt.imshow(heatmap.cpu().numpy()[0])



            # plt.subplot(141)

            # plt.title('input')

            # plt.imshow(image.cpu().numpy()[0][0])

            # plt.savefig(f'results/{cur_iter}_img.png')

            # plt.close()



            gt = sample['gt']

            A = []

            A_size = []

            for it in gt[0]:

                if it[2]==0:

                    break

                A.append([it[0].item(), it[1].item()])

                A_size.append(it[2].item())

            B = []

            B_size = []

            for x, y in zip(location[0], location[1]):

                B.append([y/128*512, x/128*512])

                B_size.append(heatmap_size.cpu().numpy()[0, x, y])



            row_ind, col_ind = best_match(np.array(A), np.array(B))

            m, n = len(A), len(B)



            relation = []

            if m > n:

                for i, j in zip(row_ind, col_ind):

                    if j < n:

                        relation.append([i, j])

            elif m < n:

                row_ind = row_ind[0:m]

                col_ind = col_ind[0:m]

                for i, j in zip(row_ind, col_ind):

                    relation.append([i, j])

            else:

                for i, j in zip(row_ind, col_ind):

                    relation.append([i, j])



            size_different = 0

            different = 0

            count = 0

            for x, y in relation:

                dis = np.sum((np.array(A[x]) - np.array(B[y]))**2)

                dis_size = (A_size[x] - B_size[y]) ** 2

                different += dis

                size_different += dis_size.item()

                count += 1

                ct += 1

                diff += dis

                diff_size += dis_size.item()



            print(f'scene {cur_iter}')

            print(f'average_different: {(different/count)**0.5},')

            print(f'average_size_different: {(size_different / count)**0.5},')

            miss += (m-n)**2

            cur_iter += 1



    print(f'checkpoint {model_name}:')

    print(f'average_different: {(diff/ct)**0.5} each object')

    print(f'average_size_different: {(diff_size/ct)**0.5} each object')

    print(f'Count RMSE {(miss/cur_iter)**0.5} ')

    print(f'{(diff/ct)**0.5}')

    print(f'{(diff_size/ct)**0.5}')

    print(f'{(miss/cur_iter)**0.5}')

    print()

