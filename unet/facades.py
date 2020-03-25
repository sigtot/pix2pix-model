import os
from torch.utils import data
from PIL import Image


def getItems(path):
    items = os.listdir(path)
    items.sort()
    return [os.path.join(path, items[i]) for i in range(len(items))]


class Facades(data.Dataset):
    def __init__(self, path):
        masks_dir = os.path.join(path, 'masks')
        targets_dir = os.path.join(path, 'target')
        self.masks = getItems(masks_dir)
        self.targets = getItems(targets_dir)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        x = self.masks[index]
        mask = Image.open(self.masks[index])
        target = Image.open(self.targets[index])
        return [mask, target]
