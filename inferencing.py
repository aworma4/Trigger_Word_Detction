'''
Create 12/05/23
Inferencing using the model and a user recorded clip
'''


# import required libraries
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import time 
import torchaudio
from IPython.display import Audio
#play clip
import torch
import torchaudio.transforms as T

def record_clip(duration,max_length = 10, folder = 'user_generated_recordings/',sample_rate = 16000,   channels = 1,out_name ='app_recording.wav'):
    '''
    Function to record clip - must be 10 secs or less
    max_length - of clip size
    folder - locatoin to save clip
    sample_rate - sr of the clip - must be the same as the model (defaut 16k)
    channels - don't change this - defaults to 1 channel 
'''
    
    #ensure duration of clip is not larger than the max clip length that can be fed into the model
    if duration >max_length:
        duration = max_length
    
    rec_length = int(duration * sample_rate)  # clip length in array size not seconds
    
    # Start recorder with the given values
    # of duration and sample frequency
    recording = sd.rec(rec_length,
                   samplerate=sample_rate, channels=channels)

    # Record audio for the given number of seconds

    time.sleep(duration)

    # This will convert the NumPy array to an audio
    # file with the given sampling frequency

    
    #save clip
    file = folder+ out_name
    write(file, sample_rate, recording)
    
    print(f'{file} saved as {duration} sec clip of size {len(recording)}')
          

        
        
    
def play_clip(file ='user_generated_recordings/app_recording.wav'):
    '''
    Play a given audio clip
    '''
    
    waveform, rsr  = torchaudio.load(file)
    return(Audio(waveform.numpy()[0], rate=rsr,autoplay=True ))

def pad_clip(file ='user_generated_recordings/app_recording.wav', offset =0,duration =10):
    '''
    Pad clip so it is the correct length to convert into a spectrogram to insert into the model
    for inferencing 
    '''
    
    
    waveform,rsr = torchaudio.load(file)    

    size_array = duration*rsr
    padded_waveform = torch.zeros(1,size_array)

    l_waveform = waveform.shape[1]
    padded_waveform[:,offset:offset+l_waveform] = waveform

    print(f'{waveform.shape} padded to {padded_waveform.shape}, clip inserted in position {offset}')

    return padded_waveform, rsr


def create_background(background_filepath =  'speech_commands_v0.02.tar/_background_noise_/doing_the_dishes.wav',sample_time =10,resample_rate=16000):
    '''
    Adapted from generate_training_data.py - loads the default background sound - clips it to 10 seconds - ready to be combined with the padded recorded clip
    '''
    #first load in background file
    b_waveform, b_sample_rate = torchaudio.load(background_filepath)

    #adjust length of background file - to extract the first self.sample_time seconds
    units = int(b_sample_rate * sample_time)
    b_waveform_t = b_waveform[0,0:units].reshape([1,units])

    
    
    #resample
    transform = torchaudio.transforms.Resample(orig_freq=b_sample_rate, new_freq=resample_rate)
    
    b_waveform_t_resample = transform(b_waveform_t)

    return b_waveform_t_resample


    

def apply_spectrogram(padded_waveform,number_frequencies = 151, number_time_steps = 400, sample_rate = 16000):
    '''
    Number frequency and time step - derived from those used in the original model createion
    '''
    

    spectrogram =  T.MelSpectrogram(sample_rate,
                                    n_fft=number_time_steps,
                                    n_mels = number_frequencies,
                                    normalized = True)

    spec = spectrogram(padded_waveform)

    print(f'Return Spectrogram of shape {spec.shape}')
    
    return spec



# print realtime detection

def real_time_detection(label_out,cut_off =0.8,    real_time = False):
    '''
    cut_off -  the probability above which a time step must be for the system to be triggered
    
    '''

    label_out_bool = torch.where(label_out > cut_off, 1.0, 0.0)

    grad = torch.gradient(label_out_bool[0])



    # Loop through label_out - delaying 
    spec_time = label_out.shape[-1]
    time_fraction = 10/spec_time

    #functionality not implimented yet to stop multiple chimes 
    delay= False 


    I_lag = 0
    for I in range(0,spec_time):

        if real_time == True:
            time.sleep(time_fraction)

        t_last= int(I_lag*time_fraction)
        t_now = int(I*time_fraction) 
        #print(t_last,t_now)

        if label_out[0,I] > 0.8 and delay == False:
            print(f'Chime - trigger word detected at {I * time_fraction} seconds, index {I}')

            delay = True

        if grad[0][I] > 0 and delay == True:
            #set delay back to false
            print(f'Chime - trigger word un detected detected at {I * time_fraction} seconds, index {I}')
            delay =False



        if  t_last < t_now:
            print(I *time_fraction , ' seconds')

        if I>0:
            I_lag = I    

    print(I *time_fraction , ' seconds')    




'''
#example usage
record_clip(duration=2)




play_clip(file ='user_generated_recordings/app_recording.wav')    

padded_waveform, rsr = pad_clip(file ='user_generated_recordings/app_recording.wav')

background = create_background()

#can weight the background sound - won't for now
bg_weight = 0.1
padded_waveform =padded_waveform + bg_weight*background

#play padded audio clip
Audio(padded_waveform.numpy()[0], rate=rsr,autoplay=False )

#convert to spectrogram
padded_spec = apply_spectrogram(padded_waveform,sample_rate =rsr)


#load model and perform inferencing
model = torch.load('model_test_mel_100_epochs')
label_out = model(padded_spec)


#graph label over spectrogram
# now graph 
plot_spectrogram(padded_spec[0],label_out.detach().numpy(), title=None, ylabel='freq_bin', aspect='auto', xmax=None)

real_time_detection(label_out,real_time=False) 


'''