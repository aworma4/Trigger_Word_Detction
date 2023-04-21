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
    
    
    def generate_positive_sample(self,background,filepath):
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
        label = self.create_labels(True,start, end, background_length)   

        return waveform_padded_random, label
    
    


        
        
    def generate_all(self):
        positive_filepaths = self.generate_filepaths(self.folder_trigger_word)
        negative_filepaths = self.generate_filepaths(self.folder_negative_word)
        
        train_outfile = self.outpath_location + '/train/'
        test_outfile = self.outpath_location + '/test/'
        
        train_I = int(np.ceil(self.test_train_split * self.number_samples ))

        #need to geneate background each loop - some how it was getting overwritten
        
        #geneate positive lables 
        for I in range(self.number_samples):
            background = self.create_background()
            positive_sample,label = self.generate_positive_sample(background,positive_filepaths[I])
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
        for I in range(self.number_samples):
            background = self.create_background()
            negative_sample,label = self.generate_positive_sample(background,negative_filepaths[I])
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
    
background_filepath =  'speech_commands_v0.02.tar/_background_noise_/doing_the_dishes.wav'
sample_time = 10
resample_rate = 5000
number_samples = 10
folder_trigger_word = 'speech_commands_v0.02.tar/nine/'
folder_negative_word = 'speech_commands_v0.02.tar/backward/'
outpath_location = 'data'

val = Create_Test_Train_Data(background_filepath, sample_time,resample_rate,number_samples, folder_trigger_word,folder_negative_word,outpath_location)

'''
