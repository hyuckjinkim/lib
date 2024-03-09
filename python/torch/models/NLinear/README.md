# NLinear : Time-Series SOTA Model
- [SOTA](https://paperswithcode.com/sota)

- Example
```python

class NLinear_Model(nn.Module):
    def __init__(self, input_size, output_size, sequence_size):
        super(NLinear_Model, self).__init__()
        super().__init__()
        
        # nn.BatchNorm1d(nodes[1]) , 
        nodes = [24]*4
        dropout = 0.00
        
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.gelu = nn.GELU()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout(dropout)
        
        self.activation = self.leakyrelu
        
        self.model = nn.Sequential(
            NLinear(sequence_size ,nodes[0]), self.dropout, nn.BatchNorm1d(nodes[0]), self.activation,
            NLinear(nodes[0],nodes[1]), self.dropout, nn.BatchNorm1d(nodes[1]), self.activation,
            NLinear(nodes[1],nodes[2]), self.dropout, nn.BatchNorm1d(nodes[2]), self.activation,
            NLinear(nodes[2],nodes[3]), self.dropout, nn.BatchNorm1d(nodes[3]), self.activation,
            NLinear(nodes[3],output_size),
        )

        self.fc   = nn.Linear(input_size,1)
        self.fc   = TimeDistributed(self.fc)
        #self._reinitialize()

        # for name, p in self.named_parameters():
        #     print(name, 'scinet' in name)
        
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

    def forward(self,x):
        x = self.model(x)
        x = self.fc(x[:,:,-1])
        return x
```