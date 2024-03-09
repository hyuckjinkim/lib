# SCINet : Time-Series SOTA Model
- [SOTA](https://paperswithcode.com/sota)

- Example
```python
from ..utils import TimeDistributed
from scinet import SCINet
from scinet_decompose import SCINet_decompose

class SCINet_Model(nn.Module):
    def __init__(self,input_size,output_size,sequence_size):
        super(SCINet_Model, self).__init__()
        super().__init__()
        
        window_size = sequence_size # in (fixed)
        horizon = 1                 # out
        hidden_size = 2
        groups = 1
        kernel = 2
        dropout = 0.0
        single_step_output_One = False
        num_levels = 1
        positionalEcoding = True
        num_stacks = 1
        
        self.scinet = SCINet(
            output_len = horizon, input_len = window_size, input_dim = input_size, hid_size = hidden_size, 
            num_stacks = num_stacks, num_levels = num_levels, concat_len = 0, groups = groups, kernel = kernel, 
            dropout = dropout, single_step_output_One = single_step_output_One, positionalE =  positionalEcoding, 
            modified = True, RIN = True,
        )
        self.scinet_decompose = SCINet_decompose(
            output_len = horizon, input_len = window_size, input_dim = input_size, hid_size = hidden_size, 
            num_stacks = num_stacks, num_levels = num_levels, concat_len = 0, groups = groups, kernel = kernel, 
            dropout = dropout, single_step_output_One = single_step_output_One, positionalE =  positionalEcoding, 
            modified = True, RIN = True,
        )
        
        self.dropout = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(horizon)
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.fc   = nn.Linear(input_size,output_size)
        self.fc   = TimeDistributed(self.fc)
        
        #self.nlinear = NLinear(input_size,1)
        self._reinitialize()

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
    
    def forward(self, x):
        x = self.scinet(x)
        # x = self.scinet_decompose(x)
        x = self.bn(x)
        x = self.fc(x[:,-1,:])
        return x
```