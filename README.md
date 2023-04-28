# AIG
Artifical Intelligence Group - project - investigation trigger word detector 

to dos:
* Add in simpler RNN and CNN models (+ test)
* SpecAugment - add random masks to frequency/time - according to reserach this helps improve performance 
* Switch to usin ga mel spectrogram - might be better .

Test values for spec_freq, spec_time, label_time  - might try setting spec_time == label_time 


* create system for takign short wave recordings (10sec <) and then apply the mdoel to then return an indication if the tw detector was activated. 

* Change initial training data to include 0-4 words random (non overlapping) with some trigger words and some not.  - should improve the utility of the tw detector
