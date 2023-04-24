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
    
