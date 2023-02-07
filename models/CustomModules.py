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
        dense_size = int((receptive_field / 2) * embedding_dim)
        print(dense_size)

        self.embedding_dim = embedding_dim
        # To keep track of KL divergence
        self.kl = 0

        self.prelinear = nn.Linear(dense_size, 512)
        self.postlinear = nn.Linear(latent_dim, dense_size)
        # To sample the latent vector
        # self.N = torch.distributions.Normal(0, 1)
        
        self.mean = nn.Linear(dense_size, latent_dim)
        self.var = nn.Linear(dense_size, latent_dim)

    def forward(self, x):
        input_shape = x.shape
        input_device = x.get_device()
        
        # self.N.loc = self.N.loc.to(input_device) # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.to(input_device)
        
        if self.verbose:
            print("##################")
            print("LatConv input size: ", input_shape)
            
        x = torch.flatten(x, start_dim=1)
        
        if self.verbose:
            print("LatConv flatten size: ", x.size())

        mu = self.mean(x)
        logvar = self.var(x)
        sigma = torch.exp(logvar/2)
        
        if self.verbose:
            print("LatConv mean size: ", mu.size())
            print("LatConv sigma size: ", sigma.size())
            
        # sampled = self.N.sample(mu.shape)
        epsilon = torch.randn_like(sigma).to(self.device)      # sampling epsilon  
        
        # print("Input device: ", input_device)
        # print("Mu device: ", mu.get_device())
        # print("Sigma device: ", sigma.get_device())
        # print("Sampled device: ", sampled.get_device())
        # print("##################")

        z = mu + sigma * epsilon
        
        if self.verbose:
            print("LatConv z size: ", z.size())
            
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        
        if self.verbose:
            print("LatConv kl: ", self.kl)
        
        z = nn.functional.relu(self.postlinear(z))
        
        if self.verbose:
            print("LatConv postlinear size: ", z.size())

        x = z.view(input_shape)
        
        if self.verbose:
            print("LatConv view size: ", x.size())
            
        return x, mu, logvar

    


class DoubleConvStack(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DoubleConvStack, self).__init__()

        convlayer_1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            padding = padding
        )

        convlayer_2 = nn.Conv2d(
            in_channels = out_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            padding = padding
        )

        self.convlayers = nn.Sequential(
            convlayer_1,
            convlayer_2
        )

    def forward(self, x):
        return self.convlayers(x)


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