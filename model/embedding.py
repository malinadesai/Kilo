class VICRegLoss(nn.Module):
    ''' 
    Variance-Invariance-Covariance Regularization Loss Function
    citation: A. Bardes, J. Ponce, and Y. LeCun. Vicreg: 
              Variance-invariance-covariance regularization 
              for self-supervised learning, 2022. 
    '''
    def forward(self, x, y, wt_repr=1.0, wt_cov=1.0, wt_std=1.0):
        repr_loss = F.mse_loss(x, y)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        N = x.size(0)
        D = x.size(-1)
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        x = (x-x.mean(dim=0))/std_x
        y = (y-y.mean(dim=0))/std_y
        # transpose dims 1 and 2; keep batch dim i.e. 0, unchanged
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        cov_x = (x.transpose(1, 2) @ x) / (N - 1)
        cov_y = (y.transpose(1, 2) @ y) / (N - 1)
        cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(D)
        cov_loss += self.off_diagonal(cov_y).pow_(2).sum().div(D)
        s = wt_repr*repr_loss + wt_cov*cov_loss + wt_std*std_loss
        return s, repr_loss, cov_loss, std_loss

    def off_diagonal(self,cov):
        num_batch, n, m = cov.shape
        assert n == m
        # All off diagonal elements from complete batch flattened
        # import pdb; pdb.set_trace()
        return cov.flatten(start_dim=1)[...,:-1].view(num_batch, n - 1, n + 1)[...,1:].flatten()
        
class ConvResidualBlock(nn.Module):
    def __init__(self, 
                 channels, 
                 kernel_size, 
                 activation=F.relu, 
                 dropout_probability=0.1, 
                 use_batch_norm=True, 
                 zero_initialization=True):
        '''
        Inputs:
            channels: number of separate dimensions, here it is photometric bands
            kernel_size: (odd int) input field size of the CNN
            activation: (function) calculates node output
            dropout_probability: (float) probability an element will be zeroed
            use_batch_norm: (bool) determines if data will be normalized by re-scaling and re-centering
            zero_initialization: (bool) initializes weights with zeros
        '''
        super().__init__()
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList([nn.BatchNorm1d(channels, eps=1e-3) for _ in range(2)])
        self.conv_layers = nn.ModuleList([nn.Conv1d(channels, channels, kernel_size=kernel_size, padding='same') for _ in range(2)])
        self.dropout = nn.Dropout(p=dropout_probability)
        if zero_initialization:
            nn.init.uniform_(self.conv_layers[-1].weight, -1e-3, 1e-3)
            nn.init.uniform_(self.conv_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs):
        temps = inputs
        if self.use_batch_norm:
            temps = self.batch_norm_layers[0](temps)
        temps = self.activation(temps)
        temps = self.conv_layers[0](temps)
        if self.use_batch_norm:
            temps = self.batch_norm_layers[1](temps)
        temps = self.activation(temps)
        temps = self.dropout(temps)
        temps = self.conv_layers[1](temps)
        return inputs + temps

class ConvResidualNet(nn.Module):
    ''' Convolutional Neural Network composed of many Convolutional Residual Blocks'''
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 hidden_channels, 
                 num_blocks, 
                 kernel_size, 
                 activation=F.relu, 
                 dropout_probability=0.1, 
                 use_batch_norm=True):
        '''
        Inputs:
            in_channels: starting number of channels, ie number of photometric bands
            out_channels: end number of channels
            hidden_channels: number of hidden channels in 1d conv. net
            num_blocks: number of conv. residual blocks to use
            kernel_size: (odd int) input field size of the CNN
            activation: (function) calculates node output
            dropout_probability: (float) probability of resetting weights
            use_batch_norm: (bool) determines if batch is normalized
        '''
        super().__init__()
        self.hidden_channels = hidden_channels
        self.initial_layer = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=kernel_size, padding='same')
        self.blocks = nn.ModuleList(
            [ConvResidualBlock(
                channels=hidden_channels, 
                activation=activation, 
                dropout_probability=dropout_probability, 
                use_batch_norm=use_batch_norm, 
                kernel_size=kernel_size) 
             for _ in range(num_blocks)])
        self.final_layer = nn.Conv1d(hidden_channels, out_channels, kernel_size=kernel_size, padding='same')

    def forward(self, inputs):
        temps = self.initial_layer(inputs)
        for block in self.blocks:
            temps = block(temps)
        outputs = self.final_layer(temps)
        return outputs

class SimilarityEmbedding(nn.Module):
    '''
    A fully connective neural network with a ResNet layer f  and an expander layer h
    '''
    def __init__(self, num_dim=3, num_hidden_layers_f=1, num_hidden_layers_h=1, num_blocks=4, kernel_size=5, num_dim_final=10, activation=torch.tanh):
        super(SimilarityEmbedding, self).__init__()
        self.layer_norm = nn.LayerNorm([num_channels, num_points])
        self.num_hidden_layers_f = num_hidden_layers_f
        self.num_hidden_layers_h = num_hidden_layers_h
        self.layers_f = ResNet(num_ifos=[3,None], layers=[2,2], kernel_size=kernel_size, context_dim=100)
        self.contraction_layer = nn.Linear(in_features=100, out_features=num_dim)
        # self.layers_f = ConvResidualNet(in_channels=num_channels, out_channels=1, hidden_channels=20, num_blocks=num_blocks, kernel_size=kernel_size)
        # self.contraction_layer = nn.Linear(in_features=in_features, out_features=num_dim)
        self.expander_layer = nn.Linear(num_dim, 20)
        self.layers_h = nn.ModuleList([nn.Linear(20, 20) for _ in range(num_hidden_layers_h)])
        self.final_layer = nn.Linear(20, num_dim_final)
        self.activation = activation
        
    def forward(self, x):
        x = self.layers_f(x)
        x = self.contraction_layer(x)
        representation = torch.clone(x)
        x = self.activation(self.expander_layer(x))
        for layer in self.layers_h:
            x = layer(x)
            x = self.activation(x)
        x = self.final_layer(x)
        return x, representation
