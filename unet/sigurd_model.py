from .art_net import ArtNet
from .pavel_net import PavelNet

class SigurdModel(object):
    def __init__(self):
        self.discriminator = ArtNet()
        self.generator = PavelNet()

        self.real_A = None
        self.real_B = None
        self.fake_B = None

    def set_input(self, real_A, real_B):
        self.real_A = real_A
        self.real_B = real_B

    def forward(self):
        self.fake_B = self.generator.forward(self.real_A)

    def backward(self):
        self.forward()