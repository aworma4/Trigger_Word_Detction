'''
Data Loading and Analysis
'''

import torchaudio.transforms as T
import matplotlib.pyplot as plt
import librosa
import torch


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

