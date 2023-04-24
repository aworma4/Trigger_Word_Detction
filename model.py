'''
# Main model file 
'''


def get_default_device():
    '''
    Pick GPU is it is avaiable, else CPU
    '''
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

    
    
import torch.nn as nn

#class to store Conv parameters - generally these won't be changed - but I'm leaving the option open here
# I'd only really want to change out_channels
class Conv_p:
    def __init__(self,kernel_size =15,stride=4, out_channels = 1):
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = 1

        
class GRU_p:
    def __init__(self, num_layers =2, batch_first = True, dropout = 0.8):
        if num_layers == 1:
            dropout = 0 # can't have a non zero dropout in the final layer
        
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        
        

        
        



class TriggerWord_LSTM(nn.Module):
    '''
    LSTM neural network for performing trigger word detection - based on paper Trigger_Word_Recognition_using_LSTM
    '''
    
    def __init__(self, input_freq, input_time , hidden_time, output_time, Conv_p, GRU_p):
        super().__init__()
        '''
        Create layers of the neural network - note freq/time denote the sizes of the 1st and 2nd dimensions respectively
        '''
        #save parameters here
        self.input_freq = input_freq
        self.input_time = input_time
        self.hidden_time = hidden_time
        self.output_time = output_time
        self.Conv_p = Conv_p
        self.GRU_p = GRU_p
        
        
        #CONV1D
        self.Conv = nn.Conv1d(in_channels = input_freq, 
                              out_channels = Conv_p.out_channels,
                              kernel_size = Conv_p.kernel_size, 
                              stride=Conv_p.stride)

        
        #GRU
        # calculate size of final dimension from conv1d - equation from documentation
        #https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        Conv_outsize = int(((input_time + 2 * self.Conv.padding[0] - self.Conv.dilation[0] * (self.Conv.kernel_size[0] -1) -1 )/self.Conv.stride[0]) + 1 )
        
        self.GRU = nn.GRU(input_size =Conv_outsize, 
        hidden_size =hidden_time,
        num_layers  = GRU_p.num_layers, 
        batch_first = GRU_p.batch_first, 
        dropout     = GRU_p.dropout)
                
        # DENSE
        self.Dense = nn.Linear(in_features = hidden_time , out_features = output_time)

        # Sigmoid layer
        self.Sigmoid = nn.Sigmoid()

        
    def forward(self,xb):
        '''
        Apply the layers to the batch input
        '''
        out = self.Conv(xb)
              
        out, hidden_state = self.GRU(out)

        out = self.Dense(out).squeeze(1)  #remove 1 singleton dimension - not the batch dimension

        out = self.Sigmoid(out)
               
        return out
        
    
def get_accuracy(y_true, y_prob,cutoff=0.8):
    y_true = y_true.squeeze()
    y_prob = y_prob.squeeze()
    
    
    assert y_true.ndim == 1 and y_true.size() == y_prob.size()
    y_prob = y_prob > cutoff
    return (y_true == y_prob).sum().item() / y_true.size(0)