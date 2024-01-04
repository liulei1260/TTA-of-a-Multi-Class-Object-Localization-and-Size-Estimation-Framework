from torch.utils.data import Dataset
from PIL import Image
import torch
import os
import albumentations as A
import numpy as np

class FruitDataset(Dataset):
    def __init__(self, data_root, is_train=False):
        super(FruitDataset, self).__init__()
        if is_train:
            dataset = 'train_data'
        else:
            dataset = 'test_data'
        self.is_train = is_train
        self.classes = os.listdir(data_root)
        self.images = []
        self.labels = []
        for c in self.classes:
            c_images = os.listdir(os.path.join(data_root, c, dataset))
            for img in c_images:
                self.images.append(os.path.join(data_root, c, dataset, img))
                self.labels.append(os.path.join(data_root, c, 'ground_truth', img.split(".")[0].split("_")[-1]+'.txt'))

        self.transform = A.Compose([A.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                                             translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, rotate=(-15, 15),
                                             shear={"x": (-10, 10), "y": (-10, 10)}),
                                    A.ColorJitter(),
                                    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0, )],
                                   keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']))

        self.norm = A.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5), max_pixel_value=255.0,)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = Image.open(self.images[idx])
        im = np.array(im)
        label = self.labels[idx]

        f = open(label)
        lines = f.readlines()
        f.close()
        line = ""
        for i in lines:
            line += i.replace('\n', '')
        lines = line[2:-2].split("},{")

        objects = {}
        for i in lines:
            l = i.split(",")
            ins = {}
            for item in l:
                key, value = item.split(":")
                if key[1:-1] == 'label':
                    ins[key[1:-1]] = value[3:-1]
                elif key[1:-1] == 'Size':
                    ins[key[1:-1]] = float(value[2:-1])
                elif key[1:-1] == 'id':
                    ins[key[1:-1]] = int(value)
                else:
                    ins[key[1:-1]] = float(value)
            objects[ins['id']] = ins

        w, h = im.shape[:2]
        heatmap = np.zeros((int(w/4), int(h/4)))
        heatmap_size = np.zeros((int(w / 4), int(h / 4)))

        keypoints = [(objects[key]['x'], objects[key]['y']) for key in objects]
        class_labels = [objects[key]['label'] for key in objects]
        radius = [int(objects[key]['Size']) for key in objects]
        radiu_float = [objects[key]['Size'] for key in objects]
        if self.is_train:
            aug = self.transform(image=im, keypoints=keypoints, class_labels=class_labels)
            im = aug['image']
            keypoints = aug['keypoints']
            class_labels = aug['class_labels']
        else:
            im = self.norm(image=im)['image']

        for pt, rd, rd_f in zip(keypoints, radius, radiu_float):
            draw_umich_gaussian(heatmap, pt, rd)
            draw_umich_gaussian_size(heatmap_size, pt, rd, rd_f)
        
        gt = []
        for key, value in objects.items():
            gt.append([value['x'], value['y'], value['Size']])

        #如果gt的len不足300，则补充0,300默认是最大的gt数量
        if len(gt) < 300:
            gt.extend([[0,0,0]] * (300 - len(gt)))

        return {'image': torch.from_numpy(im.transpose((2, 0, 1))),
                'heatmap': torch.from_numpy(heatmap).float(),
                'heatmap_size': torch.from_numpy(heatmap_size).float(),
                'gt': torch.from_numpy(np.array(gt))}

#用来生成一个 2D 高斯分布的滤波核，给定形状和 sigma 参数 
#Used to generate a filter kernel for a 2D Gaussian distribution, given the shape and sigma parameters
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0    # 限制最小的值 Minimum value
    return h

#绘制尺寸热力图 Draw size heat map
def draw_umich_gaussian_size(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1  # 直径 diameter
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 2)

    x, y = int(center[0]/4), int(center[1]/4)

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

#绘制位置热力图 Draw the location heat map
def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1  # 直径 diameter
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]/4), int(center[1]/4)

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
