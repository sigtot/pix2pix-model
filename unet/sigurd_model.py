import torch

from .art_net import ArtNet
from .pavel_net import PavelNet


class SigurdModel(object):
    def __init__(self):
        self.nn = None
        self.discriminator = ArtNet()
        self.generator = PavelNet()

        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters())
        self.opt_generator = torch.optim.Adam(self.generator.parameters())

        self.real_x = None
        self.real_y = None
        self.fake_y = None

    def set_input(self, real_x, real_y):
        self.real_x = real_x
        self.real_y = real_y

    def forward(self):
        self.fake_y = self.generator.forward(self.real_x)

    def backward(self):
        # generate "fake" image from real mask
        fake_xy = torch.cat((self.real_x, self.fake_y), 1)

        pred_fake = self.discriminator(fake_xy)
        # for training the discriminator, the prediction of the fake image should be True
        loss_discriminator_fake = self.nn.MSELoss(pred_fake, False)

        self.forward()
        real_xy = torch.cat((self.real_x, self.real_y), 1)
        pred_real = self.discriminator(real_xy)
        # for training the discriminator, the prediction of the real image should be False
        loss_discriminator_true = self.nn.MSELoss(pred_real, True)

        loss_discriminator = 0.5 * (loss_discriminator_fake + loss_discriminator_true)

        self.opt_discriminator.zero_grad()
        loss_discriminator.backward()
        self.opt_discriminator.step()

        # for training the generator, the prediction of the fake image should be True
        loss_generator = self.nn.MSELoss(pred_fake, True)

        self.opt_generator.zero_grad()
        loss_generator.backward()
        self.opt_generator.step()
