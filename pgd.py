from matplotlib import image
import numpy as np
import json

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.utils
from torchvision import models
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import argparse

# from zmq import device
from mnist import MNet

MODEL_PATH = './mnist_cnn.pt'
FIG_PATH = './figure/pgd/'

transform = transforms.Compose([
    transforms.ToTensor(),
])

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")


def pgd_attack(model, images, labels, eps=0.3, alpha=0.01, iters=40):
    images = images.to(device)
    labels = labels.to(device)
    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = F.nll_loss(outputs, labels)
        cost.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images

def imshow(img, title):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)), cmap='Greys', interpolation='nearest')
    plt.title(title)
    plt.savefig(FIG_PATH+'img_'+title)


parser = argparse.ArgumentParser(description='PyTorch MNIST PGD Attack')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
args = parser.parse_args()
train_kwargs = {'batch_size': args.batch_size}
test_kwargs = {'batch_size': args.test_batch_size}


dataset1 = datasets.MNIST('./mnist/', train=True, download=True,
                          transform=transform)
dataset2 = datasets.MNIST('./mnist/', train=False,
                          transform=transform)
train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
print("loading data complete")


model = MNet()
model.load_state_dict(torch.load(MODEL_PATH))
if torch.cuda.is_available():
    model.cuda()
model.eval()
print("loading model complete")

correct = 0
total = 0
test_loss = 0
print("start attacking test")

lbs = []
adv_imgs = []

for idx, (images, labels) in enumerate(test_loader):
    # if idx != 0:
    #     break
    # imshow(torchvision.utils.make_grid(images[0:5], normalize=True), "ori")
    images = pgd_attack(model, images, labels)
    # imshow(torchvision.utils.make_grid(images.cpu().data[0:5], normalize=True), "adv")
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    test_loss += F.nll_loss(outputs, labels, reduction='sum').item()

    pred = outputs.argmax(dim=1, keepdim=True)
    lbs.append(pred)
    correct += pred.eq(labels.view_as(pred)).sum().item()

    total += 1
    print("step:{} Accuracy:{:.2f}%".format(total, 100 * float(correct) / (total * len(images))))
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss / len(test_loader.dataset), correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
fig, ax = plt.subplots(
    nrows=2,
    ncols=5,
    sharex=True,
    sharey=True, )

img_iter = iter(test_loader)
images, labels = img_iter.next()
ax = ax.flatten()
for i in range(10):
    img = images[i][0].reshape(28, 28)
    label = lbs[i][0].data.item()
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title("labels:{}".format(label))
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.savefig(FIG_PATH+'adv')
plt.show()
#
#
# print('Accuracy of Attacking text: %.2f%%' % (100 * float(correct) / len(test_loader.dataset)))
# with torch.no_grad():
#     for data, target in test_loader:
#         data = data.to(device)
#         target = target.to(device)
#         output = model(data)
#         test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch losss
#         pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#         correct += pred.eq(target.view_as(pred)).sum().item()
#
# test_loss /= len(test_loader.dataset)
#
# print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
#     test_loss, correct, len(test_loader.dataset),
#     100. * correct / len(test_loader.dataset)))