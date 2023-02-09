from torchvision.transforms import transforms
from PIL import ImageFilter
import random

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2023, 0.1994, 0.2010)

def stats(dataset):
    if dataset == 'imgnet':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
    elif dataset == 'inat':
        mean=[0.466, 0.471, 0.380]
        std=[0.195, 0.194, 0.192]
    return mean, std

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

def get_transform(dataset, loss_fn, split):
    mean, std = stats(dataset)
    
    if loss_fn in ['CE', 'LDAM', 'BS', 'RIDE']:
        if dataset == 'imgnet':
            train_before = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0),
            ]
            
        elif dataset == 'inat':
            train_before = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
            
        else:
            raise NotImplementedError
            
        train_after = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
            
        transform_train = [[transforms.Compose(train_before), transforms.Compose(train_after)]]
    
    elif loss_fn in ['NCL', 'BCL']:
        regular_train_before = [
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0),
        ]

        regular_train_after = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        
        sim_cifar_before = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        
        sim_cifar_after = [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        
        if loss_fn == 'NCL':
            transform_train = [
                [transforms.Compose(regular_train_before), 
                transforms.Compose(regular_train_after)], 
                [transforms.Compose(sim_cifar_before), 
                transforms.Compose(sim_cifar_after)],
            ]
        else:
            transform_train = [
                [transforms.Compose(regular_train_before), 
                transforms.Compose(regular_train_after)], 
                [transforms.Compose(sim_cifar_before), 
                transforms.Compose(sim_cifar_after)], 
                [transforms.Compose(sim_cifar_before), 
                transforms.Compose(sim_cifar_after)],
            ]
        
    transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    return transform_train if split == 'train' else transform_val