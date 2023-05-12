# AIG
Artifical Intelligence Group - project - investigation trigger word detector 

Data set from Tensorflow speech_commands:
http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz


* create system for takign short wave recordings (10sec <) and then apply the mdoel to then return an indication if the tw detector was activated. 
* Functionalise and create app


* Test values for spec_freq, spec_time, label_time  - might try setting spec_time == label_time 



* Change initial training data to include 0-4 words random (non overlapping) with some trigger words and some not.  - should improve the utility of the tw detector


# Run streamlit app
streamlit run .\streamlit_app\streamlit_app.py