import torch.nn as nn

# https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, output_size, sequence_size, hidden, dropout, num_layers, bidirectional):

        assert isinstance(hidden,list) and len(hidden)==1, \
            "hidden must by type list of length 1"
        assert isinstance(dropout,list) and len(dropout)==1, \
            "dropout must by type list of length 1"
        assert isinstance(num_layers,list) and len(num_layers)==1, \
            "num_layers must by type list of length 1"
        
        if bidirectional:
            offset = 2
        else:
            offset = 1
        
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden[0],
            dropout=dropout[0],
            num_layers=num_layers[0],
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        self.elu  = nn.ELU()
        
        self.bn = nn.BatchNorm1d(sequence_size)
        
        self.activation = self.leakyrelu
        
        self.fc = nn.Linear(offset * hidden[0], output_size)
        self.fc = TimeDistributed(self.fc)
        self._reinitialize()
        
    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        # 1st
        x, _ = self.lstm(x)
        x    = self.bn(x)
        x    = self.activation(x)
        # fully connected layer
        x    = self.fc(x[:,-1,:])
        #x    = self.fc(x)
        return x
    
    
# https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization
class MultipleLSTM(nn.Module):
    def __init__(self, input_size, output_size, sequence_size, hidden, dropout, num_layers, bidirectional):

        assert isinstance(hidden,list) and len(hidden)==4, \
            "hidden must by type list of length 4"
        assert isinstance(dropout,list) and len(dropout)==4, \
            "dropout must by type list of length 4"
        assert isinstance(num_layers,list) and len(num_layers)==4, \
            "num_layers must by type list of length 4"
        
        if bidirectional:
            offset = 2
        else:
            offset = 1
        
        super().__init__()
        
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden[0],
            dropout=dropout[0],
            num_layers=num_layers[0],
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.lstm2 = nn.LSTM(
            input_size=offset*hidden[0],
            hidden_size=hidden[1],
            dropout=dropout[1],
            num_layers=num_layers[1],
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.lstm3 = nn.LSTM(
            input_size=offset*hidden[1],
            hidden_size=hidden[2],
            dropout=dropout[2],
            num_layers=num_layers[2],
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.lstm4 = nn.LSTM(
            input_size=offset*hidden[2],
            hidden_size=hidden[3],
            dropout=dropout[3],
            num_layers=num_layers[3],
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        self.elu  = nn.ELU()
        
        self.bn = nn.BatchNorm1d(sequence_size)
        
        self.activation = self.leakyrelu
        
        self.fc = nn.Linear(offset * hidden[3], output_size)
        self.fc = TimeDistributed(self.fc)
        self._reinitialize()
        
    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

    def forward(self, x):
        # 1st
        x, _ = self.lstm1(x)
        x    = self.bn(x)
        x    = self.activation(x)
        # 2nd
        x, _ = self.lstm2(x)
        x    = self.bn(x)
        x    = self.activation(x)
        # 3rd
        x, _ = self.lstm3(x)
        x    = self.bn(x)
        x    = self.activation(x)
        # 4th
        x, _ = self.lstm4(x)
        x    = self.bn(x)
        x    = self.activation(x)
        # fully connected layer
        x    = self.fc(x[:,-1,:])
        #x    = self.fc(x)
        return x