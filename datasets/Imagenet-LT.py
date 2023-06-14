import torchvision.transforms as transforms
from torch.utils.data import Dataset
import os
from PIL import Image

# Dataset
class ImageLoader(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):

        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def get_cls_num_list(self):
        from collections import Counter
        self.cls_num = 1000
        self.num_per_cls_dict = Counter(self.labels)
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

# Load datasets
def get_dataset(conf):
    datadir = "./data/ILSVRC2015/Data/CLS-LOC/"

    if conf and 'datadir' in conf:
        datadir = conf.datadir

    train_txt = '.' + '/' + 'datasets/ImageNet_LT/ImageNet_LT_train.txt'
    eval_txt = '.' + '/' + 'datasets/ImageNet_LT/ImageNet_LT_test.txt'

    conf['num_class'] = 1000
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ds_train = ImageLoader(datadir, train_txt, transform=transform_train)
    ds_test = ImageLoader(datadir, eval_txt, transform=transform_test)

    return ds_train,ds_test

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_class', type=int, default=200, help="class numbers")

    conf = parser.parse_args()
    ds_train,ds_test = get_dataset(conf)
    print(len(ds_train),len(ds_test))