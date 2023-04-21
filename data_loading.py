'''
Data Loading and Analysis
'''

import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
import torch
import glob


def create_spectrogram(waveform,number_frequencies = 101,number_time_steps = 511):
    '''
    Output is a spectrogram of size [1,number_frequencies, ~number_time_steps]
    '''
    

    n_fft = number_frequencies*2  -1 # to ensure we have 101 frequencies 
    win_length = None
    hop_length =int( waveform.shape[1]/number_time_steps) # ensures we have 5556 time steps - as close to 5511 as I could get

    # define transformation
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )
    # Perform transformation
    spec_waveform = spectrogram(waveform)
    
    print("spec_waveform.shape :", spec_waveform.shape)
    print("waveform.shape : " ,waveform.shape)
    
    return spec_waveform



def plot_spectrogram(spec,label, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    '''
    Plots a spectrogram with the corresponding labelling
    '''
    fig, ax = plt.subplots(1, 1,sharex=True)
    axs = ax
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(librosa.power_to_db(spec), origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)

    t_step = spec.shape[1]
    label_t_step = label.shape[1]
    
    x = torch.linspace(0,label.shape[1],label.shape[1]) * t_step / label_t_step  
    val = label.reshape([label.shape[1]])*100
    
    #add box showing location of the audio
    axs.plot(x,val,'r')
    axs.legend(['Trigger Word Label'])
    
    
    
    plt.show(block=False)

    
'''
Example usage

#load data
waveform, sample_rate = torchaudio.load('data/test/data_positive_7.wav')
label = torch.load('data/test/label_positive_7.pt')


spec_waveform = create_spectrogram(waveform)

    
plot_spectrogram(spec_waveform[0], label, title='torchaudio')

Audio(waveform.numpy()[0], rate=resample_rate)

'''

'''
###############################################
Custom DATA loader
###############################################
'''



class ReadData(torch.utils.data.Dataset):
    '''
    created data reader
    '''
    
    def __init__(self, str_type = 'test',spectrogram_str ='False',number_frequencies = 101,number_time_steps = 511):
          

        if str_type=='test':
            folder = 'data/test/'
        else:
            folder = 'data/train/'

        #generate file paths     

        data_negative = sorted(glob.glob(folder + 'data_negative*.wav'))
        data_positive = sorted(glob.glob(folder + 'data_positive*.wav'))

        label_negative = sorted(glob.glob(folder + 'label_negative*.pt'))
        label_positive = sorted(glob.glob(folder + 'label_positive*.pt'))

        self.data = data_negative + data_positive 
        self.label = label_negative + label_positive
        
        self.number_frequencies = number_frequencies
        self.number_time_steps = number_time_steps
        self.spectrogram_str = spectrogram_str
        
    def create_spectrogram(self,waveform_shape):
        '''
        Output is a spectrogram of size [1,number_frequencies, ~number_time_steps]
        '''


        n_fft = self.number_frequencies*2  -1 # to ensure we have 101 frequencies 
        win_length = None
        hop_length =int( waveform_shape/number_time_steps) # ensures we have 5556 time steps - as close to 5511 as I could get

        # define transformation
        spectrogram = T.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
        )
        return spectrogram    
        
        
        
    def __len__(self):
        
        return len(self.data)
    
    
    
    
    
    def __getitem__(self, index):
        # print(data)
        # print(index)
        wave,sample_rate = torchaudio.load(self.data[index])
        label= torch.load(self.label[index])
        
  
        
        #reshape to [1,x] size
        shape = label.shape[1]
        label.reshape([1,shape])
        
        
        if self.spectrogram_str == 'True':
            wave_shape = wave.shape[1]    
            spectrogram = self.create_spectrogram(wave_shape)
            wave = spectrogram(wave)
            return [wave,label]
        
        else:
            return [[wave,sample_rate],label]
    
'''
#example data loading

test = ReadData()

[[wave1,sample_rate],label1] = test[0]

[[wave2,sample_rate],label2] = test[1]


spec_wave1 = create_spectrogram(wave1)

    
plot_spectrogram(spec_wave1[0], label1, title='torchaudio')

Audio(waveform.numpy()[0], rate=resample_rate)


spec_wave2 = create_spectrogram(wave2)

    
plot_spectrogram(spec_wave2[0], label2, title='torchaudio')

Audio(waveform.numpy()[0], rate=resample_rate)

'''



'''
Create DataLoader - can specify as wav or spectrogram

from torch.utils.data import DataLoader
test = ReadData('test')
train = ReadData('train')

batch_size = 4
train_loader = DataLoader(train, batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size, shuffle=False)

'''