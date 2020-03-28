import torch
import torch.nn as nn
from torch.autograd import Variable

from .art_net import ArtNet
from .pavel_net import PavelNet


class SigurdModel(nn.Module):
    def __init__(self, lr, betas, weight_decay, disc_mult, l1_mult, l1_only):
        super(SigurdModel, self).__init__()
        self.discriminator = ArtNet()
        self.generator = PavelNet()

        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=weight_decay,
                                                  betas=betas)
        self.opt_generator = torch.optim.Adam(self.generator.parameters(), lr=lr, weight_decay=weight_decay,
                                              betas=betas)

        self.discriminator.apply(self.init_weights)
        self.generator.apply(self.init_weights)

        self.disc_mult = disc_mult
        self.l1_mult = l1_mult
        self.l1_only = l1_only
        self.loss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def init_weights(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, real_x, real_y):
        self.real_x = real_x
        self.real_y = real_y
        self.generated = self.generator.forward(self.real_x)
        return self.generated

    def backward(self):
        if self.l1_only:
            loss_generator = self.l1_mult * self.l1loss.forward(self.generated, self.real_y)
            self.opt_generator.zero_grad()
            loss_generator.backward()
            self.opt_generator.step()
            return

        pred_fake = self.discriminator(torch.cat((self.real_x, self.generated), 1))

        loss_generator = self.loss(pred_fake, torch.ones_like(pred_fake, device=self.device,
                                                              requires_grad=False)) + self.l1_mult * self.l1loss.forward(
            self.generated, self.real_y)

        self.opt_generator.zero_grad()
        loss_generator.backward()
        self.opt_generator.step()

        pred_fake = self.discriminator(torch.cat((self.real_x, self.generated), 1).detach())
        pred_real = self.discriminator(torch.cat((self.real_x, self.real_y), 1))
        loss_discriminator = self.disc_mult * (
                self.loss(pred_fake, torch.zeros_like(pred_fake, device=self.device, requires_grad=False))
                + self.loss(pred_real, torch.ones_like(pred_real, device=self.device, requires_grad=False))
        )

        self.opt_discriminator.zero_grad()
        loss_discriminator.backward()
        self.opt_discriminator.step()

        return loss_generator.data, loss_discriminator.data
