# AIG
Artifical Intelligence Group - project - investigation trigger word detector 

# Data set
Data set from Tensorflow speech_commands:
http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

# to dos 

* functionalise training + introduce MLflow 

* Re run for more training dat (currently only using 20 total)

* Change initial training data to include 0-4 words random (non overlapping) with some trigger words and some not.  - should improve the utility of the tw detector


# Run streamlit app
terminal:
streamlit run .\streamlit_app\streamlit_app.py

# Launch MLflow tracker
terminal:
mlflow server