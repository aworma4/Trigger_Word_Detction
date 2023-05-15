'''
'''




###  Model set up + hyperparmeters 



import torch
#hyper parameters 
import torch.optim as optim
from model import get_accuracy
from model import *
from data_loading import resize_label
from data_loading import *
from torch.utils.data import DataLoader
from torchsummary import summary
import numpy as np


# Create Params dictionary

class Params(object):

    def __init__(self, batch_size, test_batch_size, number_frequencies, number_time_steps, epochs, lr, seed, cuda, log_interval,early_stopper_patience,early_stopper_min_delta, label_time,cutoff):
        '''
        Names self explanatory - seed = Random seed number, log_interval - the intervals at which the weights and biases will be recorded 
        '''

        self.batch_size = batch_size

        self.test_batch_size = test_batch_size

        self.epochs = epochs

        self.lr = lr
        
        self.number_frequencies = number_frequencies
        
        self.number_time_steps = number_time_steps
        

        self.seed = seed

        self.cuda = cuda

        self.log_interval = log_interval

        self.early_stopper_patience = early_stopper_patience
        
        self.early_stopper_min_delta = early_stopper_min_delta
        
        self.label_time = label_time
        
        self.cutoff = cutoff
        
        
        
#### set devices  
args =Params(batch_size = 4, test_batch_size = 4,
             number_frequencies = 151,
             number_time_steps = 400,
             epochs = 100, lr =0.01, 
             seed = 1, cuda = False, 
             log_interval = 200,
             early_stopper_patience = 3,
             early_stopper_min_delta=0.01,
             label_time = 1375,
            cutoff =0.8) #enter zero then label_time (shape of output of output of nerual network will be equal to the size of the input spectrogram)),

cuda = not args.cuda and torch.cuda.is_available()
device = get_default_device() #not sure if I'm going to use this 
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


'''
####  Load DAta 
'''


#### Load data 

test_waveform = ReadData_Mel('test',spectrogram_str=False)
test_data = ReadData_Mel('test',spectrogram_str=True, normalize = True, mask_str = False, number_frequencies = args.number_frequencies,number_time_steps = args.number_time_steps, t_l = 100, f_l = 2)
#ReadData('test',spectrogram_str='True',number_frequencies = spec_freq,number_time_steps = spec_time)
train_data = ReadData_Mel('train',spectrogram_str=True, normalize = True, mask_str = False, number_frequencies = args.number_frequencies,number_time_steps = args.number_time_steps, t_l = 100, f_l = 2)
#ReadData('train',spectrogram_str='True',number_frequencies = spec_freq,number_time_steps = spec_time)

train_loader = DataLoader(train_data, args.batch_size, shuffle=False)
test_loader = DataLoader(test_data, args.batch_size, shuffle=False)



#### Initialise model

# import the model and initialise 
#load in to know the size of the spectrogram - the fft does quite give you a spec_time length for the time dimension
spec_wave = test_data[-1][0]
spec_label = test_data[-1][1]



input_freq =spec_wave.shape[1]
input_time = spec_wave.shape[2]

output_time = args.label_time
if output_time ==0:
    output_time = input_time


hidden_time = output_time #could change this down the line 


model =TriggerWord_LSTM(input_freq, input_time , hidden_time, output_time, Conv_p(),GRU_p())
 
#torch.load('model_test_28_04_23_100_epochs')
#TriggerWord_LSTM(input_freq, input_time , hidden_time, output_time, Conv_p(),GRU_p())
#SimpleRNN(time = input_time,dropout_rate=0.5)

#model.Conv
print(f"input shape: (1,{input_freq},{input_time})")  
summary(model, (input_freq,input_time))     



#### Early stopping condition 
#create early stopper class
#I'll be using training instead of validation loss 
#https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    
#### Choose optimizers,scheduler, loss, and early stopping condtion
criterion = torch.nn.BCELoss()

#alternate optmizer + scheduler - from guidance from https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/
optimizer = optim.AdamW(model.parameters(), args.lr)

scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
	max_lr=args.lr,
	steps_per_epoch=int(len(train_data)),
	epochs=args.epochs,
	anneal_strategy='linear')

early_stopper = EarlyStopper(patience=args.early_stopper_patience, min_delta=args.early_stopper_min_delta)



'''
Main training functions
'''

if cuda:
    model.cuda()

writer = None # Will be used to write TensorBoard events

def log_scalar(name, value, step):

    """Log a scalar value to both MLflow and TensorBoard"""

    #writer.add_summary(scalar(name, value).eval(), step)

    mlflow.log_metric(name, value, step=step)
    
    
    

def train(epoch,N_trainloader):
    running_loss = 0.0
    running_accuracy = 0.0
    running_prec = 0.0
    running_rec = 0.0
    running_av_label = 0.0 
    model.train()
    for i, data in enumerate(train_data):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels_og = data
        
        labels =  resize_label(labels_og, args.label_time)
        

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        if early_stopper.early_stop(loss):             
            break
        
        
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_accuracy += f_acc(outputs,labels)
        #get_accuracy(labels, outputs,cutoff=args.cutoff)
        running_prec += f_prec(outputs,labels)
        running_rec  += f_rec(outputs,labels)
        running_av_label += torch.mean(labels)
        # if i % 4 == 3:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.7f}')
        #     running_loss = 0.0
    
    lr_end = scheduler.get_last_lr()
    acc = running_accuracy/N_trainloader
    prec = running_prec/N_trainloader
    rec = running_rec/N_trainloader
    
    mean_label = running_av_label/N_trainloader
    
    print(f'Epoch [{epoch + 1}] loss: {loss}, learning rate {lr_end}, training accuracy cutoff ({args.cutoff}): {acc}, average label {mean_label}')
        
    log_scalar('loss',loss,step=epoch)
    log_scalar('lr',lr_end[0],step=epoch)
    log_scalar('accuracy_with_cutoff',acc,step=epoch)
    log_scalar('precision_with_cutoff',prec,step=epoch)
    log_scalar('recall_with_cutoff',rec,step=epoch)    

    log_scalar('mean_label',mean_label,step=epoch)
    
    scheduler.step()


    
    
    
def test(epoch,test_data):
    running_loss = 0.0
    running_accuracy = 0.0
    running_prec = 0.0
    running_rec = 0.0
    running_av_label = 0.0 
    
    model.eval()
    
    N_trainloader = len(test_data)


    for i, data in enumerate(test_data):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels_og = data

        labels =  resize_label(labels_og, args.label_time)

        outputs = model(inputs)


        running_accuracy += f_acc(outputs,labels)
        #get_accuracy(labels, outputs,cutoff=args.cutoff)
        running_prec += f_prec(outputs,labels)
        running_rec  += f_rec(outputs,labels)
        running_av_label += torch.mean(labels)

    acc = running_accuracy/N_trainloader
    prec = running_prec/N_trainloader
    rec = running_rec/N_trainloader    
    mean_label = running_av_label/N_trainloader
    print(f'Test for Epoch [{epoch + 1}], accuracy  = {acc}, precision =  {prec}, recall =  {rec}, average label {mean_label}')

    

    
    
def main():
    N_trainloader = len(train_data)

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        
        train(epoch,N_trainloader)
        #test(epoch,test_data)

    

    
    scheduler.step()    
    
    
    

import mlflow.pytorch
from torchmetrics.classification import BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy

#F1 = BinaryF1Score(threshold=args.cutoff)
f_prec = BinaryPrecision(threshold=args.cutoff)
f_rec = BinaryRecall(threshold=args.cutoff)
f_acc = BinaryAccuracy(threshold=args.cutoff)


#choose experiment name
experiment_name = 'test1'
mlflow.set_experiment(experiment_name)
experiment = mlflow.get_experiment_by_name(experiment_name)

with mlflow.start_run() as run:  
    # Log our parameters into mlflow
    for key, value in vars(args).items():
        mlflow.log_param(key, value)
    
    main()