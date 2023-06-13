

import sys

import numpy

import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from conf import settings
from torch.utils.data.distributed import DistributedSampler
def get_network(args):
    """ return given network
    """
    if args.net == 'Aresnet18':
        from models.A_ResNet import resnet18
        net = resnet18()
    elif args.net == 'Aresnet34':
        from models.A_ResNet import resnet34
        net = resnet34()
    elif args.net == 'Sresnet18':
        from models.S_ResNet import resnet18
        net = resnet18(args.dataset)
    elif args.net == 'Sresnet34':
        from models.S_ResNet import resnet34
        net = resnet34(args.dataset)                      
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()

    return net

def get_teacher_network(args):
    """ return given network
    """
    if args.teacher_net == 'resnet18':
        from models.ResNet import resnet18
        net = resnet18(args.dataset)
    elif args.teacher_net == 'resnet34':
        from models.ResNet import resnet34
        net = resnet34(args.dataset)                  
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net

def get_network_CIFAR(args):
    if args.net == 'Sresnet18':
        from models.S_ResNet import resnet18_CIFAR
        net = resnet18_CIFAR()
    return net


def get_training_dataloader(traindir, sampler=None, batch_size=16, num_workers=2, shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ImageNet_training = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))
    if sampler is not None:
        ImageNet_training_loader = DataLoader(
            ImageNet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True, sampler=DistributedSampler(ImageNet_training))
    else:
        ImageNet_training_loader = DataLoader(ImageNet_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,pin_memory=True)

    return ImageNet_training_loader

def get_test_dataloader(valdir, sampler=None, batch_size=16, num_workers=2, shuffle=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ImageNet_test = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256), #320
            transforms.CenterCrop(224),#288
            transforms.ToTensor(),
            normalize,
        ]))
    if sampler is not None:
        ImageNet_test_loader = DataLoader(
            ImageNet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True, sampler=DistributedSampler(ImageNet_test))
    else:
        ImageNet_test_loader = DataLoader(
            ImageNet_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
        

    return ImageNet_test_loader


def get_training_dataloader_CIFAR(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), sampler=None,batch_size=16, num_workers=2, shuffle=True, dataset = 'cifar100'):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    # cifar100_training = torchvision.datasets.CIFAR100(root=settings.CIFAR_DATA_PATH, train=True, download=True, transform=transform_train)
    # if sampler is not None:
    #     cifar100_training_loader = DataLoader(
    #         cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True, sampler=DistributedSampler(cifar100_training))
    # else:
    #     cifar100_training_loader = DataLoader(
    #         cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    if dataset == 'cifar100':
        cifar100_training = torchvision.datasets.CIFAR100(root=settings.CIFAR100_DATA_PATH, train=True, download=True, transform=transform_train)
        if sampler is not None:
            cifar_training_loader = DataLoader(
                cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True, sampler=DistributedSampler(cifar100_training))
        else:
            cifar_training_loader = DataLoader(
                cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif dataset == 'cifar10':
        cifar10_training = torchvision.datasets.CIFAR10(root=settings.CIFAR10_DATA_PATH, train=True, download=True, transform=transform_train)
        print("This is cifar10!!!!!!!!!\n")
        if sampler is not None:
            cifar_training_loader = DataLoader(
                cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True, sampler=DistributedSampler(cifar10_training))
        else:
            cifar_training_loader = DataLoader(
                cifar10_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return cifar_training_loader

def get_test_dataloader_CIFAR(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761), sampler=None, batch_size=16, num_workers=2, shuffle=True, dataset = 'cifar100'):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # cifar100_test = torchvision.datasets.CIFAR100(root=settings.CIFAR_DATA_PATH, train=False, download=True, transform=transform_test)
    # if sampler is not None:
    #     cifar100_test_loader = DataLoader(
    #         cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True, sampler=DistributedSampler(cifar100_test))
    # else:
    #     cifar100_test_loader = DataLoader(
    #         cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    if dataset == 'cifar100':
        cifar100_test = torchvision.datasets.CIFAR100(root=settings.CIFAR100_DATA_PATH, train=False, download=True, transform=transform_test)
        if sampler is not None:
            cifar_test_loader = DataLoader(
                cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True, sampler=DistributedSampler(cifar100_test))
        else:
            cifar_test_loader = DataLoader(
                cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif dataset == 'cifar10':
        cifar10_test = torchvision.datasets.CIFAR10(root=settings.CIFAR10_DATA_PATH, train=False, download=True, transform=transform_test)
        print("This is cifar10!!!!!!!!!\n")
        if sampler is not None:
            cifar_test_loader = DataLoader(
                cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True, sampler=DistributedSampler(cifar10_test))
        else:
            cifar_test_loader = DataLoader(
                cifar10_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    return cifar_test_loader
