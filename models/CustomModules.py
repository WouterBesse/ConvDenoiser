from torch import nn
import torch

def getMinMax(sp_min, sp_max, total_min, total_max):
    total_min = total_min
    total_max = total_max
    
    if sp_min < total_min:
        total_min = sp_min
    if sp_max > total_max:
        total_max = sp_max
        
    return total_min, total_max


class VariationalLatentConverter(nn.Module):

    def __init__(self, receptive_field, hidden_dim, embedding_dim, latent_dim, device, verbose = False):
        super(VariationalLatentConverter, self).__init__()

        self.device = device
        self.verbose = verbose
        self.dense_size = int((receptive_field / 2) * embedding_dim)
        print("Dense size:", self.dense_size)

        self.embedding_dim = embedding_dim
        # To keep track of KL divergence
        self.kl = 0

        # self.prelinear = nn.Linear(self.dense_size, 512)
        self.postlinear = nn.Linear(latent_dim, self.dense_size)
        # To sample the latent vector
        # self.N = torch.distributions.Normal(0, 1)
        
        self.mean = nn.Linear(self.dense_size, latent_dim)
        self.var = nn.Linear(self.dense_size, latent_dim)
        
    def bottleneck(self, h):
        mu, logvar = self.mean(h), self.var(h)
        if self.verbose:
            print("LatConv mean size: ", mu.size())
            print("LatConv logvar size: ", logvar.size())
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        eps = torch.randn(*mu.size()).to(mu.get_device())
        z = mu + std * eps
        return z

    def forward(self, x):
        input_shape = x.size()
        input_device = x.get_device()
        
        # self.N.loc = self.N.loc.to(input_device) # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.to(input_device)
        
        if self.verbose:
            print("##################")
            print("LatConv input size: ", input_shape)
            
        x = x.view(x.size(0), -1)
        # print("View out:",torch.min(x), torch.max(x), torch.isnan(x).sum().item())
        if self.verbose:
            print("LatConv flatten size: ", x.size())

        z, mu, logvar = self.bottleneck(x)
        
        # z = self.reparameterize(mu, logvar) 
        
        
        
        if self.verbose:
            print("LatConv z size: ", z.size())
       
        
        z = self.postlinear(z)
        # print("Z out:",torch.min(z), torch.max(z), torch.isnan(z).sum().item())
        
        if self.verbose:
            print("LatConv postlinear size: ", z.size())

        # x = z.view(input_shape)
        x = z.view(input_shape)
        
        if self.verbose:
            print("LatConv view size: ", x.size())
            
        return x, mu, logvar

    


class DoubleConvStack(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DoubleConvStack, self).__init__()

        self.convlayer_1 = nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            padding = padding
        )

        self.convlayer_2 = nn.Conv1d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            padding = padding
        )
        
        self.leaky = nn.LeakyReLU()

        # self.convlayers = nn.Sequential(
        #     convlayer_1,
        #     convlayer_2
        # )

    def forward(self, x):
        x1 = self.convlayer1(x)
        x2 = self.convlayer2(x)
        x = x1 + x2
        
        self.leaky(x)
        
        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        
        relu_1 = nn.ReLU(True)
        conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_residual_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        relu_2 = nn.ReLU(True)
        conv_2 = nn.Conv1d(
            in_channels=num_residual_hiddens,
            out_channels=num_hiddens,
            kernel_size=1,
            stride=1,
            bias=False
        )

        self.resBlock = nn.Sequential(
            relu_1,
            conv_1,
            relu_2,
            conv_2
        )
    
    def forward(self, x):
        return x + self.resBlock(x)


class ResidualStack(nn.Module):

    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        
        self.num_residual_layers = num_residual_layers

        self.layers = nn.ModuleList(
            [ResidualBlock(in_channels, num_hiddens, num_residual_hiddens)] * self.num_residual_layers)
        
    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)

        return nn.functional.relu(x)