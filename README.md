# AIG
Artifical Intelligence Group - project - investigation trigger word detector 

# Data set
Data set from Tensorflow speech_commands:
http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

# to dos 

* Re run for more training data (currently only using 20 total) 
Changed to 200 total - works for recognisign the location of a word but not for the word itself - change examples to accomodate this .

* Change initial training data to include 0-4 words random (non overlapping) with some trigger words and some not.  - should improve the utility of the tw detector
Done

* run for new data set

* Change model to input batches of audio - i.e. add a 4th batch dimension

# Run streamlit app
terminal:
streamlit run streamlit_app.py

# Launch MLflow tracker
terminal:
mlflow server



#  Using the Code 

### Download Data
Download the tensorflow audio recordings http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz


### Data Creation
Adjust the example in generate_training_data for the required number of training/test data sets.


### Training 
Run the train_func.py - may need to change file paths + paramters etc
jupyter notebook:
%run train_func.py

### Inferencing
Load streamlit app - point app at the torch saved model - record your own audios
streamlit run streamlit_app.py