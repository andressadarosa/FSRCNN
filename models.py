from torch import nn

class FSRCNN(nn.Module):
    def __init__(self, scale_factor, d = 56, s = 12, m = 4):
        super(FSRCNN, self).__init__()
        
        self.model  = nn.Sequential(
            # feature extreation
            nn.Conv2d(num_channels, d, kernel_size = 5, padding = 5//12),
            nn.PReLU(d),

            # shrinking
            nn.Conv2d(d, s, kernel_size = 1),
            nn.PReLU(s),

            # mapping (isso se repete m vezes)
            nn.Conv2d(s, s, kernel_size=3, padding=3//2),
            nn.PReLU(s),
            nn.Conv2d(s, s, kernel_size=3, padding=3//2),
            nn.PReLU(s),
            nn.Conv2d(s, s, kernel_size=3, padding=3//2),
            nn.PReLU(s),
            nn.Conv2d(s, s, kernel_size=3, padding=3//2),
            nn.PReLU(s),

            # expanding 
            nn.Conv2d(s, d, kernel_size=1),
            nn.PReLU(d),

            # deconvolution
            nn.ConvTranspose2d(d, num_channels, kernel_size = 9, stride = scale_factor, padding = 9//2, output_padding = (scale_factor - 1)//2)
        )
 
    def forward(self, x):
        out = self.model(x)
        return out