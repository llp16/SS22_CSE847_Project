import torchvision


def load_data(dataset, train):
    if dataset == 'mnist':
        train_data = torchvision.datasets.MNIST(
            root='./'+dataset+'/',
            train=train,  # this is training data
            transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
            # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
            download=False,  # download it if you don't have it
        )
    else:
        train_data = torchvision.datasets.CIFAR10(
            root='./'+dataset+'/',
            train=train,  # this is training data
            transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
            # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
            download=False,  # download it if you don't have it
        )
    return train_data
