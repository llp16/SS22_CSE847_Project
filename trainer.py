import torch
import matplotlib.pyplot as plt
import encoder
import torch.utils.data as Data
import torchvision
import numpy as np
import torch.nn as nn


N_TEST_IMG = 5
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
PLOT_PATH = "./figure/"


class Trainer(object):

        def __init__(self, encoder, batch_size, epoch, model_path):
            self._batch_size = batch_size
            self._epoch = epoch
            self._encoder = encoder
            self._model_path = model_path

        def load_data(self):
            train_data = torchvision.datasets.MNIST(
                root='./mnist/',
                train=True,  # this is training data
                transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
                # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
                download=DOWNLOAD_MNIST,  # download it if you don't have it
            )
            train_loader = Data.DataLoader(dataset=train_data, batch_size=self._batch_size, shuffle=True)
            return train_data, train_loader

        def train(self):
            self._encoder.train()
            train_data, train_loader = self.load_data()

            optimizer = torch.optim.Adam(self._encoder.parameters(), lr=LR)
            loss_func = nn.MSELoss()

            # initialize figure
            f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
            # continuously plot
            plt.ion()

            # original data (first row) for viewing
            view_data = train_data.data[:N_TEST_IMG].view(-1, 28 * 28).type(torch.FloatTensor) / 255.
            for i in range(N_TEST_IMG):
                a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray');
                a[0][i].set_xticks(())
                a[0][i].set_yticks(())

            img_idx = 0

            for epoch in range(self._epoch):
                for step, (x, b_label) in enumerate(train_loader):
                    b_x = x.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)
                    b_y = x.view(-1, 28 * 28)  # batch y, shape (batch, 28*28)

                    encoded, decoded = self._encoder(b_x)

                    loss = loss_func(decoded, b_y)  # mean square error
                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

                    if step % 100 == 0:
                        print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

                        # plotting decoded image (second row)
                        _, decoded_data = self._encoder(view_data)
                        for i in range(N_TEST_IMG):
                            a[1][i].clear()
                            a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
                            a[1][i].set_xticks(())
                            a[1][i].set_yticks(())
                        plt.draw()
                        plt.savefig(PLOT_PATH+"encoder_{}".format(img_idx))
                        img_idx += 1
                        plt.pause(0.05)
            plt.ioff()
            torch.save(self._encoder.state_dict(), self._model_path)
            print("model successfully saved")
