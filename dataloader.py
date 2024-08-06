import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

torch.manual_seed(42)

# print("PyTorch Version: ", torch.__version__)
# print("Torchvision Version: ", torchvision.__version__)

# train_dir = "C:/Users/alabi/OneDrive - University of North Carolina at Charlotte/Personal Projects/Machine Learning/ASDID/Datasets/Soybean_ML_orig_20/train"
# test_dir = "C:/Users/alabi/OneDrive - University of North Carolina at Charlotte/Personal Projects/Machine Learning/ASDID/Datasets/Soybean_ML_orig_20/test"
# dir_path = "C:/Users/alabi/OneDrive - University of North Carolina at Charlotte/Personal Projects/Machine Learning/ASDID/Datasets/Soybean_ML_orig_20"
train_dir = "../Soybean_ML_orig/train"
test_dir = "../Soybean_ML_orig/test"
dir_path = "../Soybean_ML_orig"

class ImageDatasetLoader:
    def __init__(self, data_dir, batch_size, world_size, rank):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        # self.num_workers = num_workers
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.data_transforms = self._get_data_transforms()
        self.image_datasets = self._get_image_datasets()
        self.dataloaders = self._get_dataloaders(sampler=None)
        self.dataset_sizes = self._get_dataset_sizes()
        self.class_names = self.image_datasets['train'].classes
        # self.sampler = DistributedSampler(self, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False)

    def _get_data_transforms(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
                transforms.ColorJitter(brightness=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
        }
        return data_transforms

    def _get_image_datasets(self):
        sets = ['train', 'val', 'test']
        image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x]) for x in sets}
        return image_datasets
    
    def update_dataloaders(self, sampler):
        self.dataloaders = self._get_dataloaders(sampler)

    def _get_dataloaders(self, sampler):   
        dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=self.batch_size, shuffle=False, num_workers=0, pin_memory=False, sampler = sampler) for x in self.image_datasets}
        return dataloaders

    def _get_dataset_sizes(self):
        dataset_sizes = {x: len(self.image_datasets[x]) for x in self.image_datasets}
        return dataset_sizes
    
    def _imshow(self, inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = self.mean
        std = self.std
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.show()  # Show the plot and block execution until closed

    def plot_random_images(self, dataset_type='train', num_images=4):
        """Plot a batch of random images from the specified dataset type (train, val, test)."""
        dataloader = self.dataloaders[dataset_type]
        # Get a batch of training data
        inputs, classes = next(iter(dataloader))
        
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs[:num_images])

        self._imshow(out, title=[self.class_names[x] for x in classes[:num_images]])
    
    def plot_random_image(self, dataset_type='train'):
        """Plot a random image from the specified dataset type (train, val, test)."""
        dataloader = self.dataloaders[dataset_type]
        # Get a random batch of training data
        inputs, classes = next(iter(dataloader))
        
        idx = np.random.randint(len(inputs))
        img, label = inputs[idx], classes[idx]
        
        self._imshow(img, title=self.class_names[label])


# if __name__ == '__main__':
#     data_loader = ImageDatasetLoader(data_dir=dir_path, batch_size=32)
#     train_loader = data_loader.dataloaders['train']
#     val_loader = data_loader.dataloaders['val']
#     test_loader = data_loader.dataloaders['test']
#     dataset_sizes = data_loader.dataset_sizes
#     class_names = data_loader.class_names

#     data_loader.plot_random_images(dataset_type='train', num_images=4)
#     # data_loader.plot_random_image(dataset_type='val')

#     print(dataset_sizes, class_names)
