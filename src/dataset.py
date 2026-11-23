import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")  #label = index in this list


class SignCrops(Dataset):
    """Reads data/crops/<signer>/<letter>/*.jpg for each signer in `signers`."""

    def __init__(self, signers, train=False):
        self.items = []  #(image path, label index)
        for signer in signers:
            root = os.path.join("data", "crops", signer)
            for letter in sorted(os.listdir(root)):
                if letter not in LETTERS:
                    continue
                folder = os.path.join(root, letter)
                for name in sorted(os.listdir(folder)):
                    self.items.append((os.path.join(folder, name), LETTERS.index(letter)))

        steps = []
        if train:  #augmentation on the training set only
            steps += [
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            ]
        steps += [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.tf = transforms.Compose(steps)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        image = Image.open(path).convert("RGB")
        return self.tf(image), label