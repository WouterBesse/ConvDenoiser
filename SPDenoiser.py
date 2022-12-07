from torch import nn


# Torch implementation of https://github.com/sthalles/cnn_denoiser
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = 'same', use_bn=True, skip=False):
        super().__init__()

        self.conv1D = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, padding = padding)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.use_bn = use_bn
        self.skip = skip

    def forward(self, x, oldskip = None):
        skip = self.conv1D(x)
        print("Convstep size: ", skip.size())

        if oldskip is not None:
            skip = skip + oldskip

        x = self.relu(skip)
        if self.use_bn:
            x = self.bn(x)
        if self.skip:
            return x, skip
        else:
            return x

class ConvSeq(nn.Module):
    def __init__(self):
        super().__init__()

        self.Conv1 = ConvBlock(8, 18, (9,1))
        self.Conv2 = ConvBlock(18, 30, (5,1), skip=True)
        self.Conv3 = ConvBlock(30, 8, (9,1))

    def forward(self, x, skip=None):
        x = self.Conv1(x)
        x, skip = self.Conv2(x, skip)
        x = self.Conv3(x)
        return x, skip
    

# Denoiser will get tensors of 129 features (frequency) by 8 segments (timesteps)
class SPDenoiser(nn.Module):
    def __init__(self, features, segments):
        super().__init__()

        self.zeroPadding = nn.ZeroPad2d((0, 0, 4, 4))
        self.firstConv = ConvBlock(1, 18, (9,segments), padding='valid')
        self.secondConv = ConvBlock(18, 30, (5,1), skip=True)
        self.thirdConv = ConvBlock(30, 8, (9,1))

        self.ConvSeq1 = ConvSeq()
        self.ConvSeq2 = ConvSeq()
        self.ConvSeq3 = ConvSeq()
        self.ConvSeq4 = ConvSeq()
        # self.fourthConv = ConvBlock(8, 18, (9,1))
        # self.fifthConv = ConvBlock(18, 30, (5,1), skip=True)
        # self.sixthConv = ConvBlock(30, 8, (9,1))
        # self.seventhConv = ConvBlock(8, 18, (9,1))
        # self.eightConv = ConvBlock(18, 30, (5,1), skip=True)
        # self.ninthConv = ConvBlock(30, 8, (9,1))

        self.dropout = nn.Dropout2d(p=0.2)
        self.finalConv = nn.Conv1d(8, 1, kernel_size=(features, 1), bias=False, padding = 'same')

        
    def forward(self, input):
        x = self.zeroPadding(input)
        x = self.firstConv(x)
        x, skip0 = self.secondConv(x)
        x = self.thirdConv(x)
        x, skip1 = self.ConvSeq1(x)
        x, red = self.ConvSeq2(x)
        x, red = self.ConvSeq3(x, skip1)
        x, red = self.ConvSeq4(x, skip0)
        x = self.dropout(x)
        x = self.finalConv(x)
        return x
