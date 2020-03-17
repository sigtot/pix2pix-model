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

        self.real_A = None
        self.real_B = None
        self.fake_B = None

    def set_input(self, real_A, real_B):
        self.real_A = real_A
        self.real_B = real_B

    def forward(self):
        self.fake_B = self.generator.forward(self.real_A)

    def backward(self):
        # discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.discriminator(fake_AB)
        loss_discriminator_fake = self.nn.MSELoss(pred_fake, False)

        self.forward()
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.discriminator(real_AB)
        loss_discriminator_true = self.nn.MSELoss(pred_real, True)

        loss_discriminator = 0.5 * (loss_discriminator_fake + loss_discriminator_true)
        loss_discriminator.backward()

