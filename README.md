# Bioacoustics-Machine-Learning

HMM_final.py:
HMM that classifies sequences of sequences of feature vectors with kFolds. 
Parameters to change: k(number of classes), file names, kf_num (number of kfolds)

HMM_final_2files.py:
HMM that classifies with one file as train and another as test. 
Parameters to change: k(number of classes), file names, kf_num (number of kfolds)

HMM_VQ_final.py:
HMM that classifies sequences of integers in VQ vectors with kFolds. 
Parameters to change: k(number of classes), file names, kf_num (number of kfolds)

runWhaleDataVisualizer:
A mostly self-contained Matlab function that runs a visualization of spectrograms from whale data.
It requires the datasets, background spectrogram dataset, and MTRead function to run.

TODO:
Make parameters command line argument/user input to the python scripts.
