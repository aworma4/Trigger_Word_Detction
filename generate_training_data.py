'''
Contains class used to build the training data 
21/04/2023

great now have the process
* functionalise - create of training data (input for background, filepath location of trigger word/neutral word, no random seed, filepath location to save the data)
* easiest to resample the data first 
load_background(filepath)
generate_clip(resampling, background, filepath_word, outpath_location)  - save waveform, sample_rate, label all saved to file

Run over list of negative word s( 1 from each)  + list of 10 words.

'''
import torchaudio
import torch
import glob
import random #No random seed 
import numpy as np

class Create_Test_Train_Data():
    
    def __init__(self, background_filepath, sample_time,resample_rate,number_samples, folder_trigger_word,folder_negative_word,outpath_location):
        '''
        background_filepath - location of the .wave file containg the background noise
        sample_time - length of the entire clip - default 10 sec - needs to be greater than length of the negative/trigger word
        resample_rate - the number of samples persecond to reduce the data set to
        number_samples - the numberf of clips containing the trigger word - there will be an equal number not containing the trigger word
        folder_trigger_word -  folder containing the examples of the trigger word
        folder_negative_word - folder containing the negative word - for now I'm only using 1 word - could improve to have multiple words 
        outpath_location - location to save the training/test data sets
        '''
        self.background_filepath = background_filepath
        self.sample_time = sample_time
        self.resample_rate = resample_rate
        self.number_samples = number_samples
        self.folder_trigger_word = folder_trigger_word
        self.folder_negative_word = folder_negative_word
        self.outpath_location = outpath_location
        self.test_train_split = 0.7
    
    #create a resample 
    
    
    def resample(self,sample_rate,new_sample_rate,waveform):
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        transformed_waveform = transform(waveform)
        return transformed_waveform

        
    
    
    def create_background(self):
        #first load in background file
        b_waveform, b_sample_rate = torchaudio.load(self.background_filepath)
        
        #adjust length of background file - to extract the first self.sample_time seconds
        units = int(b_sample_rate * self.sample_time)
        b_waveform_t = b_waveform[0,0:units].reshape([1,units])
        
        b_waveform_t_resample = self.resample(b_sample_rate,self.resample_rate,b_waveform_t)
        
        return b_waveform_t_resample
    
    
    def generate_filepaths(self,filepath):
        temp = glob.glob(filepath + '*.wav')
        
        temp2 = temp[:self.number_samples]
        
        return temp2
    
    def create_labels(self, bool_tw,start, end, background_length):
        ''' create label for the given data set'''
          
        #create labels 
        label_random= torch.zeros([1,background_length])
    
        if bool_tw==False:
            return label_random
        else: 
            #add in the ones where the audio occurs 
            label_random[0,start:end] = torch.ones([1,end-start])
            return label_random    
    
    
    def generate_sample(self,bool_tw,background,filepath):
        # load audio  
        waveform, sample_rate = torchaudio.load(filepath)
        #resample 
        waveform_resample = self.resample(sample_rate,self.resample_rate,waveform)
        
        #Create location to insert waveform into background
        
        # add word in a rondom place
        background_length = background.shape[1]
        tw_length = waveform_resample.shape[1]

  
        start_rand = 0
        end_rand = background_length - tw_length #so that we can fit the word in 

        start = random.randint(start_rand, end_rand)
        end = start + tw_length
        print(f"inserted clip into {start} position - {start/self.resample_rate} seconds out of {self.sample_time} seconds")

        waveform_padded_random = background
        
        
 
        
        waveform_padded_random[0,start:end]  = waveform_resample + background[0,start:end]

        # finally create label
        label = self.create_labels(bool_tw,start, end, background_length)   

        return waveform_padded_random, label
    
    


        
        
    def generate_all(self):
        positive_filepaths = self.generate_filepaths(self.folder_trigger_word)
        negative_filepaths = self.generate_filepaths(self.folder_negative_word)
        
        train_outfile = self.outpath_location + '/train/'
        test_outfile = self.outpath_location + '/test/'
        
        train_I = int(np.ceil(self.test_train_split * self.number_samples ))

        #need to geneate background each loop - some how it was getting overwritten
        
        #geneate positive lables 
        bool_tw = True
        for I in range(self.number_samples):
            background = self.create_background()
            positive_sample,label = self.generate_sample(bool_tw,background,positive_filepaths[I])
            name= f'data_positive_{I}.wav'
            label_name = f'label_positive_{I}.pt'


            if I < train_I:
                path = train_outfile+name
                path_label = train_outfile+label_name

            else:
                path = test_outfile+name
                path_label = test_outfile+label_name

            #save result
            print(path)
            print(path_label)
            torchaudio.save(path,positive_sample,self.resample_rate)  
            torch.save(label,path_label)
            
        #generate negative labels
        bool_tw = False
        for I in range(self.number_samples):
            background = self.create_background()
            negative_sample,label = self.generate_sample(bool_tw,background,negative_filepaths[I])
            name= f'data_negative_{I}.wav'
            label_name = f'label_negative_{I}.pt'


            if I < train_I:
                path = train_outfile+name
                path_label = train_outfile+label_name

            else:
                path = test_outfile+name
                path_label = test_outfile+label_name

            #save result
            print(path)
            print(path_label)
            torchaudio.save(path,negative_sample,self.resample_rate)  
            torch.save(label,path_label)        
        
        return positive_sample, negative_sample, 
    
'''    
# run the class     

nine folder > 3k files
backward folder > 1.5k files

    
background_filepath =  'speech_commands_v0.02.tar/_background_noise_/doing_the_dishes.wav'
sample_time = 10
resample_rate = 5000 #think I want to change this to 16k - the original rate
number_samples = 10
folder_trigger_word = 'speech_commands_v0.02.tar/nine/'
folder_negative_word = 'speech_commands_v0.02.tar/backward/'
outpath_location = 'data'

val = Create_Test_Train_Data(background_filepath, sample_time,resample_rate,number_samples, folder_trigger_word,folder_negative_word,outpath_location)

val.generate_all()
'''



'''
Next iteration of the data generator

'''
import torch
import random
from random import choice, seed
import numpy as np

import torchaudio
import glob


class Create_Test_Train_Data_multiple_recs():
    
    def __init__(self, max_clips,background_filepath, sample_time,resample_rate,number_samples, folder_trigger_word,folder_negative_word,outpath_location):
        '''
        max_clips - max number of clips of negative or postive tip featuring in the data - for now total number of clips will be 2*max_clips as 
        we will take an even amount form both negative and positive
        background_filepath - location of the .wave file containg the background noise
        sample_time - length of the entire clip - default 10 sec - needs to be greater than length of the negative/trigger word
        resample_rate - the number of samples persecond to reduce the data set to
        number_samples - the numberf of clips containing the trigger word - there will be an equal number not containing the trigger word
        folder_trigger_word -  folder containing the examples of the trigger word
        folder_negative_word - folder containing the negative word - for now I'm only using 1 word - could improve to have multiple words 
        outpath_location - location to save the training/test data sets
        '''
        self.max_clips = max_clips
        self.background_filepath = background_filepath
        self.sample_time = sample_time
        self.resample_rate = resample_rate
        self.number_samples = number_samples
        self.folder_trigger_word = folder_trigger_word
        self.folder_negative_word = folder_negative_word
        self.outpath_location = outpath_location
        self.test_train_split = 0.7
    
    #create a resample 
    
    
    def resample(self,sample_rate,new_sample_rate,waveform):
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sample_rate)
        transformed_waveform = transform(waveform)
        return transformed_waveform

        
    
    
    def create_background(self):
        #first load in background file
        b_waveform, b_sample_rate = torchaudio.load(self.background_filepath)
        
        #adjust length of background file - to extract the first self.sample_time seconds
        units = int(b_sample_rate * self.sample_time)
        b_waveform_t = b_waveform[0,0:units].reshape([1,units])
        
        b_waveform_t_resample = self.resample(b_sample_rate,self.resample_rate,b_waveform_t)
        
        return b_waveform_t_resample
    
    
    def generate_filepaths(self,filepath):
        temp = glob.glob(filepath + '*.wav')
        
        temp2 = temp[:self.number_samples]
        
        return temp2
    
    def create_labels_list(self,label_random, bool_tw,start, end, background_length):
        ''' create label for the given data set'''
          

    
        if bool_tw==False:
            return label_random
        else: 
            #add in the ones where the audio occurs 
            label_random[0,start:end] = torch.ones([1,end-start])
            return label_random    
    
  

    def generate_clip_ranges(self,clip_lengths,Lb):
        '''
        clip_lengths - list containing the index lenght of the N clips
        Lb - length of the background - torch tensor of shape [1,Lb]

        #essentially the sum of all the deltas and sum of all the lengths of the clips must be less than the total length of the background
        # we already know the lengths
        # so below we randomly generate N-1 of the deltas (choosing values in the range (1,int(length background / (number of clips + 1)) so we can satisfy 
        # the necessary condition for this algorithm to work 
        # additionally we want a non zero delta hence we start the range at 1
        # the final delta is generated by subtracting   L background - sum(lengths) - sum(N-1 deltas) - giving a consistent answer 

        '''
        if Lb < sum(clip_lengths) +5:
            print('background too small to fit clips')
            return

        Nclips = len(clip_lengths)

        # remaining free space to use as padding around the clips
        rem = Lb - sum(clip_lengths)

        #initialise the delta variable 
        delta = [0] *Nclips

        #create the N-1 delta values
        for I in range(Nclips-1):
            delta[I] = choice([i for i in range(1,int(Lb/(Nclips+1)))])

        #calculate the final delta so that the clips can't overlap
        #so we don't gen any overlap - note its an inequality hence the use of random choice again 
        max_delta =  rem - sum(delta) 
        if max_delta == 1:
            delta[-1] = 1
        else:
            delta[-1] = choice([i for i in range(1, max_delta)])

        #print(delta)

        #now generate ranges
        start = 0
        end = 0
        clip_locs = [0]*Nclips
        for I in range(Nclips):
            a = delta[0:I+1]
            b = clip_lengths[0:I]
            start = sum(a) + sum(b)
            end = start +clip_lengths[I]
            clip_locs[I] = range(start,end)

        return delta, clip_locs     
    
    
    
    
    
    def generate_sample_list(self,bool_tw_list,background,filepath_list):

        
        # add word in a rondom place
        background_length = background.shape[1]
        print(f'background shape {background_length}')
     
        
        N_lists = len(filepath_list)
        
        clip_lengths = [0]*N_lists
        
        #generate ranges by first generating sizes of clip lengths        
        for I in range(N_lists):
            filepath = filepath_list[I]
            waveform, sample_rate = torchaudio.load(filepath)
            clip_lengths[I] = waveform.shape[1]
        print(f'clip_lengths ={clip_lengths}')
        print(background_length)
        (d,clip_ranges) = self.generate_clip_ranges(clip_lengths,Lb = background_length)
        
        
        
        waveform_padded_random = background
        #create labels 
        label= torch.zeros([1,background_length])
        
        for I in range(N_lists):
            filepath = filepath_list[I]
            bool_tw = bool_tw_list[I]
            # load audio  
            waveform, sample_rate = torchaudio.load(filepath)
            #resample 
            waveform_resample = self.resample(sample_rate,self.resample_rate,waveform)

            #Create location to insert waveform into background





            start = clip_ranges[I][0]
            end = clip_ranges[I][-1] +1
            print(f'(start/end {start}/{end}')

            print(f"inserted clip into {start} position - {start/self.resample_rate} seconds out of {self.sample_time} seconds")





            # print(waveform_resample.shape)
            # print(background.shape)
            # print(waveform_padded_random.shape)
            waveform_padded_random[0,start:end]  = waveform_resample + waveform_padded_random[0,start:end]

            # finally create label
            label = self.create_labels_list(label,bool_tw,start, end, background_length)   

        return waveform_padded_random, label
        
    
    
    

        
        
    def generate_all(self):
        positive_filepaths = self.generate_filepaths(self.folder_trigger_word)
        negative_filepaths = self.generate_filepaths(self.folder_negative_word)
        
        train_outfile = self.outpath_location + '/train/'
        test_outfile = self.outpath_location + '/test/'
        
        train_I = int(np.ceil(self.test_train_split * self.number_samples ))

        #need to geneate background each loop - some how it was getting overwritten
        
        
        
                
        #geneate positive lables 
        bool_tw_list = [False] * (2*self.max_clips)
        #need to convert the first set into false
        bool_tw_list[0:self.max_clips] = [True]* self.max_clips
        print(bool_tw_list)

        
        for I in range(self.number_samples):
            #call each loop to stop overwriting
            background = self.create_background()
        
            filepath_list = positive_filepaths[0:self.max_clips]
            temp = negative_filepaths[0:self.max_clips]
            filepath_list = filepath_list + temp

            name= f'data_positive_n_negative_N_{self.max_clips}_I_{I}.wav'
            label_name = f'label_positive_n_negative_N_{self.max_clips}_I_{I}..pt'
            print(filepath_list)


            sample, label = self.generate_sample_list(bool_tw_list,background,filepath_list)

            if I < train_I:
                path = train_outfile+name
                path_label = train_outfile+label_name

            else:
                path = test_outfile+name
                path_label = test_outfile+label_name
            
            #save result
            print(path)
            print(path_label)
            torchaudio.save(path,sample,self.resample_rate)  
            torch.save(label,path_label)
        
        
        
        return sample, label
    

    
'''
Example usage

# run the class     

from generate_training_data import *
# nine folder > 3k files
# backward folder > 1.5k files


background_filepath =  'speech_commands_v0.02.tar/_background_noise_/doing_the_dishes.wav'
sample_time = 10
resample_rate = 16000 #think I want to change this to 16k - the original rate
number_samples = 10
folder_trigger_word = 'speech_commands_v0.02.tar/nine/'
folder_negative_word = 'speech_commands_v0.02.tar/backward/'
outpath_location = 'data_multiple_clips'

#create files with 1 clip from each, two clips from each, and 1 from each
val_1 = Create_Test_Train_Data_multiple_recs(1,background_filepath, sample_time,resample_rate,number_samples, folder_trigger_word,folder_negative_word,outpath_location)
val_2 = Create_Test_Train_Data_multiple_recs(2,background_filepath, sample_time,resample_rate,number_samples, folder_trigger_word,folder_negative_word,outpath_location)

#generates 2*number_samples clips
#generates 2*number_samples clips
number_samples_half = int(number_samples/2)
val_either = Create_Test_Train_Data(background_filepath, sample_time,resample_rate,number_samples_half, folder_trigger_word,folder_negative_word,outpath_location)


wav,label_out = val_1.generate_all()
wav,label_out = val_2.generate_all()
wav,label_out = val_either.generate_all()

'''

'''
Hear audio
from IPython.display import Audio
Audio(wav.numpy()[0], rate=16000,autoplay=True )
    
''


''
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