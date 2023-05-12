import streamlit as st
from inferencing import *
from data_loading import plot_spectrogram
import matplotlib.pyplot as plt
import librosa
import torch

def st_play_clip(filename):
    audio_file = open(filename, 'rb')
    audio_bytes = audio_file.read()

    st.audio(audio_bytes, format='audio/ogg')


    
def st_plot_spectrogram(spec,label, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    '''
    Plots a spectrogram with the corresponding labelling
    Note the amplitude is rescalled so that the lower values aren't set uinformally to zero
    - using teh librosa.power_to_db()
    '''
    fig, ax = plt.subplots(1, 1,sharex=True)
    axs = ax
    axs.set_title(title or 'Spectrogram (input rescaled using librosa.power_to_db())')
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
    
    
    
    return fig    




def st_real_time_detection(label_out,cut_off =0.8,    real_time = False):
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
            st.text(f'Chime - trigger word detected at {I * time_fraction} seconds, index {I}')

            delay = True

        if grad[0][I] > 0 and delay == True:
            #set delay back to false
            st.text(f'Chime - trigger word un detected detected at {I * time_fraction} seconds, index {I}')
            delay =False



        if  t_last < t_now:
            st.text(f'{I *time_fraction} , seconds')

        if I>0:
            I_lag = I    

    st.text(f'{I *time_fraction} ,  seconds')    

    
    
    
    
    

def main():
    
    st.set_page_config(
    page_title='Trigger_Word_Detection',
    page_icon =":titanic:",
    layout="wide",
    initial_sidebar_state="expanded")

    st.title('Trigger Word Detection')
    st.sidebar.header("File Selection Options")
    
    
    duration = st.sidebar.slider('Choose length of clip to record',0.,10.,step=1.,value=2.)
    folder = st.sidebar.text_input('Recording Folder Name', 'user_generated_recordings/')
    file = st.sidebar.text_input('Recording Save Name', 'app_recording.wav')
    
    bg_weight = st.sidebar.slider('Choose weighting for background - between 0 and 1',0.,1.,step=0.01,value=0.1)
    
    model_name = st.sidebar.text_input('Model Location', 'model_test_mel_100_epochs')
    
    cut_off = st.sidebar.number_input('Model Probability Cutoff Value',0.,1.,value=0.8)
    
    filename = folder + file
    
    #example usage
    #

    if st.sidebar.button('Record Audio Clip'):
        record_clip(duration=duration, folder=folder, out_name = file)
        
        st_play_clip(filename =filename)
        st.text('Clip saved in : ' + filename)
    else:
        st.write('No Audio Clip Recorded')
        
        

    if st.sidebar.button('Inference Audio Clip'):

        
        padded_waveform, rsr = pad_clip(file =filename)

        background = create_background()

        #can weight the background sound - won't for now

        padded_waveform =padded_waveform + bg_weight*background

        #play padded audio clip
        st.text('Full padded audio clip')
        st.audio(padded_waveform.numpy()[0], sample_rate=rsr)

        #convert to spectrogram
        padded_spec = apply_spectrogram(padded_waveform,sample_rate =rsr)


        #load model and perform inferencing
        model = torch.load(model_name)
        label_out = model(padded_spec)


        #graph label over spectrogram
        # now graph 
        st.text('The graph shows the spectrogram and the value of the label probability (scaled between 0 and 100) as a function of time')
        fig = st_plot_spectrogram(padded_spec[0],label_out.detach().numpy(), title=None, ylabel='freq_bin', aspect='auto', xmax=None)
        st.pyplot(fig)
        
        
        st.text('List of the times where the system has been triggered (if at all!)')
        
        st_real_time_detection(label_out,cut_off=cut_off,real_time=False)         

        
        
        
        
    else:
        st.write('No Inferencing of Clip Started')        
        

main()


# play_clip(file =filename)    

# padded_waveform, rsr = pad_clip(file ='user_generated_recordings/app_recording.wav')

# background = create_background()

# #can weight the background sound - won't for now
# bg_weight = 0.1
# padded_waveform =padded_waveform + bg_weight*background

# #play padded audio clip
# Audio(padded_waveform.numpy()[0], rate=rsr,autoplay=False )

# #convert to spectrogram
# padded_spec = apply_spectrogram(padded_waveform,sample_rate =rsr)


# #load model and perform inferencing
# model = torch.load('model_test_mel_100_epochs')
# label_out = model(padded_spec)


# #graph label over spectrogram
# # now graph 
# plot_spectrogram(padded_spec[0],label_out.detach().numpy(), title=None, ylabel='freq_bin', aspect='auto', xmax=None)

# real_time_detection(label_out,real_time=False) 