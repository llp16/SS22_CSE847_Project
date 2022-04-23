import torchvision
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision.utils
from mnist import MNet
import loader
# from pgd import pgd_attack

class Reconstructor(object):
    def __init__(self, encoder, batch_size, epoch, dataset):
        self._encoder = encoder
        self._batch_size = batch_size
        self._epoch = epoch
        self._device = torch.device("cuda")
        # self._adv_type = adv_type
        self._dataset = dataset


    def load_data(self):
        test_data = loader.load_data(self._dataset, True)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=self._batch_size, shuffle=True)
        return test_data, test_loader

    def pgd_attack(self, model, images, labels, eps=0.3, alpha=0.01, iters=40):
        # images = images.to(device)
        # print(images)
        # labels = labels.to(device)
        images = images.to(self._device)
        labels = labels.to(self._device)
        ori_images = images.data
        loss_func = nn.MSELoss()

        # random_nosie = torch.Tensor(images.shape).uniform_(-eps, eps).to(self.device)
        # # print(random_nosie)
        # images = torch.clamp(ori_images + random_nosie, min=0, max=1).detach_()

        for _ in range(iters):
            images.requires_grad = True
            outputs = model(images)

            model.zero_grad()
            cost = F.nll_loss(outputs, labels)
            cost.backward()
            adv_images = images + alpha * images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

        return images

    def eval_attack(self):
        self._encoder.eval()

        _, test_loader = self.load_data()

        m_net = MNet()
        m_net.load_state_dict(torch.load("./model/mnist_cnn.pt"))
        if torch.cuda.is_available():
            m_net.cuda()
            self._encoder.cuda()
        m_net.eval()

        MSE_list = []

        for epoch in range(self._epoch):
            for step, (images, labels) in enumerate(test_loader):
                # b_x = images.view(-1, 28 * 28)  # batch x, shape (batch, 28*28)
                b_y = images.view(-1, 28 * 28).to(self._device)  # batch y, shape (batch, 28*28)
                # images = images.to(device)
                # labels = labels.to(device)

                att_images = self.pgd_attack(m_net, images, labels)
                b_x = att_images.view(-1, 28 * 28)
                _, decoded = self._encoder(b_x)

                loss_func = nn.MSELoss()
                loss = loss_func(decoded, b_y)  # mean square error
                MSE_list.append(loss)

                if step % 100 == 0:
                    print('Epoch: ', epoch, '| test loss: %.4f' % loss.cpu().data.numpy())
        print("Test set MSE:{:.4f}".format(torch.mean(torch.stack(MSE_list))))
        