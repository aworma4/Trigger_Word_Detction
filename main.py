'''
Minimum Working Example 
'''


'''
Data - have a data set of wav files 
with, for example, the words {nine, backwards, zero} recorded. The trigger word can be 
chosen as any of those 3 and the file structure would look like:
File stucture:
speech_commands_v0.02.tar
    _background_noise_   stores the backgroudn noise wav files - these clips are > 1 min long
    nine 
    backwards
    zero    
        -  the above 3 files contain the .wav files contain recordings of the word that has the same name as the folder
        

'''



'''
# GENERATE TRAINING DATA
Choose the filepaths to the data

Note the resample rate need to be chosen correctly to match the sample rate the .wav file swere recorded at

Outpath location - is where the data sets will be output to
'''

Example usage

# run the class     

from generate_training_data import *


background_filepath =  'speech_commands_v0.02.tar/_background_noise_/doing_the_dishes.wav'
sample_time = 10
resample_rate = 16000 #think I want to change this to 16k - the original rate
number_samples = 100
folder_trigger_word = 'speech_commands_v0.02.tar/nine/'
folder_negative_word = 'speech_commands_v0.02.tar/zero/'  #changed from backward (2 sylables) to zero
outpath_location = f'data_multiple_clips_ns_{number_samples}_negword_zero'

import os
#os.mkdir(outpath_location)
os.mkdir(outpath_location + '/train')
os.mkdir(outpath_location + '/test')


#create files with 1 clip from each, two clips from each, and 1 from each
# class description: Create_Test_Train_Data_multiple_recs(max_clips,background_filepath, sample_time,resample_rate,number_samples, folder_trigger_word,folder_negative_word,outpath_location)

#intialise classes to create the data set
val_1 = Create_Test_Train_Data_multiple_recs(1,background_filepath, sample_time,resample_rate,number_samples, folder_trigger_word,folder_negative_word,outpath_location)

val_2 = Create_Test_Train_Data_multiple_recs(2,background_filepath, sample_time,resample_rate,number_samples, folder_trigger_word,folder_negative_word,outpath_location)

#generates 1/2*number_samples clips
number_samples_half = int(2 * number_samples/2)  #changed to using the same number
val_either = Create_Test_Train_Data(background_filepath, sample_time,resample_rate,number_samples_half, folder_trigger_word,folder_negative_word,outpath_location)


wav,label_out = val_1.generate_all()
wav,label_out = val_2.generate_all()
wav,label_out = val_either.generate_all()


#Play audio using the followign code
from IPython.display import Audio
Audio(wav.numpy()[0], rate=16000,autoplay=True )

#visualise as spectrogram
import torchaudio.transforms as T
from data_loading import plot_spectrogram
def apply_spectrogram(padded_waveform,number_frequencies = 151, number_time_steps = 400, sample_rate = 16000):
   
    #Number frequency and time step - derived from those used in the original model createion
   
    

    spectrogram =  T.MelSpectrogram(sample_rate,
                                    n_fft=number_time_steps,
                                    n_mels = number_frequencies,
                                    normalized = True)

    spec = spectrogram(padded_waveform)

    print(f'Return Spectrogram of shape {spec.shape}')
    
    return spec

padded_spec = apply_spectrogram(wav)



plot_spectrogram(padded_spec[0],label_out.detach().numpy(), title=None, ylabel='freq_bin', aspect='auto', xmax=None)


'''
# Train Model
'''
#note if you want ot change the parmeters then change 
#args on likne 67 of the file - I could have used click commands for this but it was easier not to
'''
#NOTE CHANGING THE VLAUES HERE WON'T CHANGE THEM IN THE TRAIN_FUNC.PY FILE!
args =Params(batch_size = 4, test_batch_size = 4,
             number_frequencies = 151,
             number_time_steps = 400,
             epochs = 200, lr =0.01, 
             seed = 1, cuda = False, 
             log_interval = 200,
             early_stopper_patience = 5,
             early_stopper_min_delta=0.01,
             label_time = 801,#changed to same size as input time dimension of spectrogram.
            cutoff =0.1) #enter zero then label_time (shape of output of output of nerual network will be equal to the size of the input spectrogram)),
'''
from train_func import *
#the final output is a .pt file called after the 'out_name' parameter defined in train_func


'''
#Run inferencing tests
It's easiest to launch the streamlit app and point the app at the .pt file contain your trained model
The app allows the user to make recording and test from there
'''
os.system(streamlit run streamlit_app.py)