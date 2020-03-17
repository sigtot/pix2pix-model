from torch import nn


class ArtNet(nn.Module):
    def __init__(self):
        super(ArtNet, self).__init__()

        n_in = 1        # number of input channels
        n_out = 64      # number of output channels (i.e., number of filter in the last conv layer)

        self.kernel_size = 4
        self.padding = 1

        sequence = [
            nn.Conv2d(n_in, n_out, kernel_size=self.kernel_size, stride=2, padding=self.padding),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += self._get_layer(n_out, 2*n_out, 2)
        sequence += self._get_layer(2*n_out, 4*n_out, 2)
        sequence += self._get_layer(4*n_out, 8*n_out, 1)
        sequence += [
            nn.Conv2d(8*n_out, 1, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

    def _get_layer(self, n_input_channels, n_output_channels, stride):
        return [
            nn.Conv2d(n_input_channels, n_output_channels, kernel_size=self.kernel_size, stride=stride, padding=self.padding),
            nn.BatchNorm2d(n_output_channels),
            nn.LeakyReLU(0.2, True)
        ]
