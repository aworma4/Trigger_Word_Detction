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

epochs = 4




#### Load data 

test_waveform = ReadData('test',spectrogram_str='False')
test = ReadData('test',spectrogram_str='True',number_frequencies = 101,number_time_steps = 1375)
train = ReadData('train',spectrogram_str='True',number_frequencies = 101,number_time_steps = 1375)

batch_size = 4
train_loader = DataLoader(train, batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size, shuffle=False)


#### Initialise model

# import the model and initialise 
spec_wave = test[-1][0]
spec_label = test[-1][1]

spec_label_new =  resize_label(spec_label, 500)


input_freq =spec_wave.shape[1]
input_time = spec_wave.shape[2]
output_time = spec_label_new.shape[1]
hidden_time = output_time #could change this down the line 


model = TriggerWord_LSTM(input_freq, input_time , hidden_time, output_time, Conv_p(),GRU_p())


#model.Conv
summary(model, (101,1389))    

#### Choose optimizers
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)




##### Training step

trainloader = train

for epoch in range(epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels_og = data
        
        labels =  resize_label(labels_og, 500)
        

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 4 == 3:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.7f}')
            running_loss = 0.0

print('Finished Training')



##### Print accuracy for each part of the test data

# test on test data
testloader = test
for i, data in enumerate(testloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels_og = data

    labels =  resize_label(labels_og, 500)
    
    out = model(inputs)
    
    print(get_accuracy(labels, out,cutoff=0.8))
    
    #plot 
    #plot_new_vs_old_label(labels,out.detach().numpy())