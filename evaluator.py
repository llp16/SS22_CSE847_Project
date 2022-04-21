import torch
import torch.utils.data as Data
import torchvision
import torch.nn as nn


class Evaluator(object):

    def __init__(self, encoder, batch_size, epoch):
        self._encoder = encoder
        self._batch_size = batch_size
        self._epoch = epoch

    def load_data(self):
        test_data = torchvision.datasets.MNIST(
            root='./mnist/',
            train=False,  # this is training data
            transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
            # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
            download=False,  # download it if you don't have it
        )
        test_loader = Data.DataLoader(dataset=test_data, batch_size=self._batch_size, shuffle=True)
        return test_data, test_loader

    def evaluate(self):
        self._encoder.eval()

        _, test_loader = self.load_data()

        MSE_list = []

        for epoch in range(self._epoch):
            for step, (x, b_label) in enumerate(test_loader):
                b_x = x.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)
                b_y = x.view(-1, 28 * 28)  # batch y, shape (batch, 28*28)

                encoded, decoded = self._encoder(b_x)

                loss_func = nn.MSELoss()
                loss = loss_func(decoded, b_y)  # mean square error
                MSE_list.append(loss)

                if step % 100 == 0:
                    print('Epoch: ', epoch, '| test loss: %.4f' % loss.data.numpy())
        print("Test set MSE:{:.4f}".format(torch.mean(torch.stack(MSE_list))))



