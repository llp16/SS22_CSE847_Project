import torch
import torch.utils.data as Data
import torchvision
import torch.nn as nn
import loader


class Evaluator(object):

    def __init__(self, encoder, batch_size, epoch, dataset):
        self._encoder = encoder
        self._batch_size = batch_size
        self._epoch = epoch
        self._dataset = dataset
        self._img_size = 28 if dataset == 'mnist' else 32

    def load_data(self):
        test_data = loader.load_data(self._dataset, False)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=self._batch_size, shuffle=True)
        return test_data, test_loader

    def generate_noise(self, images):
        eps = 0.3
        random_nosie = torch.Tensor(images.shape).uniform_(-eps, eps)
        images = torch.clamp(images + random_nosie, min=0, max=1).detach_()
        return images

    def evaluate(self):
        self._encoder.eval()

        _, test_loader = self.load_data()

        MSE_list = []

        for epoch in range(self._epoch):
            for step, (x, b_label) in enumerate(test_loader):
                b_x = x.view(-1, self._img_size**2)  # batch x, shape (batch, 28*28)
                b_y = x.view(-1, self._img_size**2)  # batch y, shape (batch, 28*28)

                # b_x = self.generate_noise(b_x)

                encoded, decoded = self._encoder(b_x)

                loss_func = nn.MSELoss()
                loss = loss_func(decoded, b_y)  # mean square error
                MSE_list.append(loss)

                if step % 100 == 0:
                    print('Epoch: ', epoch, '| test loss: %.4f' % loss.data.numpy())
        print("Test set MSE:{:.4f}".format(torch.mean(torch.stack(MSE_list))))



