import torch
import torch.nn as nn
from torch.autograd import Variable

from .art_net import ArtNet
from .pavel_net import PavelNet


class SigurdModel(nn.Module):
    def __init__(self):
        super(SigurdModel, self).__init__()
        self.discriminator = ArtNet()
        self.generator = PavelNet()

        lr = 0.0002
        self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=1e-8,
                                                  betas=(0.5, 0.999))
        self.opt_generator = torch.optim.Adam(self.generator.parameters(), lr=lr, weight_decay=1e-8,
                                              betas=(0.5, 0.999))

        self.loss = nn.MSELoss()
        self.l1loss = nn.L1Loss()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, real_x, real_y):
        self.real_x = real_x
        self.real_y = real_y
        self.generated = self.generator.forward(self.real_x)
        return self.generated

    def backward(self):
        pred_fake = self.discriminator(torch.cat((self.real_x, self.generated), 1))
        pred_real = self.discriminator(torch.cat((self.real_x, self.real_y), 1))

        loss_generator = self.loss(pred_fake, torch.ones_like(pred_fake, device=self.device,
                                                              requires_grad=False)) + self.l1loss.forward(
            self.generated, self.real_x)

        self.opt_generator.zero_grad()
        loss_generator.backward(retain_graph=True)
        self.opt_generator.step()

        loss_discriminator = 0.5 * (
                self.loss(pred_fake, torch.zeros_like(pred_fake, device=self.device, requires_grad=False))
                + self.loss(pred_real, torch.ones_like(pred_real, device=self.device, requires_grad=False))
        )

        self.opt_discriminator.zero_grad()
        loss_discriminator.backward()
        self.opt_discriminator.step()

        return loss_generator.data, loss_discriminator.data
