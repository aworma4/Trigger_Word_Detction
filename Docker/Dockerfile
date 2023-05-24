# first layer is our python base image enabling us to run pip
FROM python:3.10-windowsservercore-1809 

# create directory in the container for adding your files
WORKDIR /user/src/AIG 

# copy over the requirements file and run pip install to install the packages into your container at the directory defined above
COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt --user 
COPY . . 

#launch jupyterlab
#RUN python -m jupyterlab

# enter entry point parameters executing the container
ENTRYPOINT ["python", "./runserver.py"] 

# exposing the port to match the port in the runserver.py file
EXPOSE 5555

# USER STILL NEEDS TO DONWLOAD TENSOR FLOW DATAset from: http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
#RUN wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
#RUN mv speech_commands_v0.01.tar.gz /user/src/speech_commands_v0.02.tar

# UNZIP - as it's windows might be easier to do it with click commands

#NOW need to clone from: - the internet advice appears to be to git clone outside of the Dockerfile
#https://github.com/aworma4/Trigger_Word_Detection.git

# THEN CREATE THE DATA SET REQUIRED for training the neural network by running code in generate_training_data.py

