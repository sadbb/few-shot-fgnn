import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

class Dataset(Dataset):
    def __init__(self, root, partition='train' ):
        data_path = os.path.join(root, partition)
        if not os.path.exists(data_path):
            raise ValueError('Wrong dataset path!') 
        label_list = os.listdir(data_path)
        label_dir = [os.path.join(data_path, label) for label in label_list if os.path.isdir(os.path.join(data_path, label))]
         
        data = []
        label = []
        for idx, label_path in enumerate(label_dir):
            img_list = os.listdir(label_path)
            for image_path in img_list:
                data.append(os.path.join(label_path, image_path))
                label.append(idx)

        self.data = data
        self.label = label
        self.num_class = len(set(label))

        mean_pix = [x / 255.0 for x in [125.3, 123.0, 113.9]]
        std_pix = [x / 255.0 for x in [63.0, 62.1, 66.7]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # Transformation
        if partition=='train':
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.RandomResizedCrop(88),
                transforms.CenterCrop(80),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(92),
                transforms.CenterCrop(80),
                transforms.ToTensor(),
                normalize])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label

class Categories_Sampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

class Categories_Sampler_v2():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        self.m_ind = torch.stack(self.m_ind, dim=0)
        self.class_num = self.m_ind.size(0)
        self.img_per_class = self.m_ind.size(1)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        img_id = [torch.randperm(self.img_per_class) for _ in range(self.class_num)]
        classes_id = torch.randperm(self.class_num)
        i, j = 0, 0
        for i_batch in range(self.n_batch):
            if i == self.class_num//self.n_cls:
                classes_id = torch.randperm(self.class_num)
                i = 0
            if j == self.img_per_class//self.n_per:
                j = 0
                img_id = [torch.randperm(self.img_per_class) for _ in range(self.class_num)]
            batch = []
            classes = classes_id[i*self.n_cls:(i+1)*self.n_cls]
            for c in classes:
                pos = img_id[c][j*self.n_per:(j+1)*self.n_per]
                l = self.m_ind[c]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            i += 1
            j += 1
            yield batch
