'''
Main training file

'''
import torch
#hyper parameters 
import torch.optim as optim
from model import get_accuracy
from model import *
from data_loading import resize_label
from data_loading import *
from torch.utils.data import DataLoader
from torchsummary import summary

epochs = 10
cutoff = 0.1 # using a really low one 
lr =0.01
batch_size = 4

#spectrogram parameters
spec_freq = 101
spec_time = 5511
#label time
label_time = 1375




#### Load data 

test_waveform = ReadData('test',spectrogram_str='False')
test = ReadData('test',spectrogram_str='True',number_frequencies = spec_freq,number_time_steps = spec_time)
train = ReadData('train',spectrogram_str='True',number_frequencies = spec_freq,number_time_steps = spec_time)

train_loader = DataLoader(train, batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size, shuffle=False)


#### Initialise model

# import the model and initialise 
#load in to know the size of the spectrogram - the fft does quite give you a spec_time length for the time dimension
spec_wave = test[-1][0]
spec_label = test[-1][1]



input_freq =spec_wave.shape[1]
input_time = spec_wave.shape[2]
output_time = label_time
hidden_time = output_time #could change this down the line 


model = TriggerWord_LSTM(input_freq, input_time , hidden_time, output_time, Conv_p(),GRU_p())


#model.Conv
print("input shape: (1,{input_freq},{input_time})")  
summary(model, (input_freq,input_time))     

#### Choose optimizers
criterion = torch.nn.BCELoss()

#alternate optmizer + scheduler - from guidance from https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch/
optimizer = optim.AdamW(model.parameters(), lr)
scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
	max_lr=lr,
	steps_per_epoch=int(len(train)),
	epochs=epochs,
	anneal_strategy='linear')

# from torch.optim.lr_scheduler import ExponentialLR 
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = ExponentialLR(optimizer, gamma=0.9)
#torch.optim.SGD(model.parameters(), lr=lr) # doesnot work



##### Training step

trainloader = train
N_trainloader = len(trainloader)

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    running_accuracy = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels_og = data
        
        labels =  resize_label(labels_og, label_time)
        

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        running_accuracy += get_accuracy(labels, outputs,cutoff=cutoff)
        # if i % 4 == 3:    # print every 2000 mini-batches
        #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.7f}')
        #     running_loss = 0.0
    
    
    print(f'Epoch [{epoch + 1}] loss: {loss}, learning rate {scheduler.get_lr()}, training accuracy cutoff ({cutoff}): {running_accuracy/N_trainloader}')
    scheduler.step()
    

print('Finished Training')

### Save model
torch.save(model, 'model_test')


##### Print accuracy for each part of the test data

# test on test data
testloader = test
for i, data in enumerate(testloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels_og = data

    labels =  resize_label(labels_og, label_time)
    
    out = model(inputs)
    
    print(get_accuracy(labels, out,cutoff=0.8))
    
    #plot 
    plot_new_vs_old_label(labels,out.detach().numpy())
    
    
   