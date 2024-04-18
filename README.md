# Marine-Animal-sound-detection
A Machine Learning Approach to Marine Animal Sound Detection and Classification 


import librosa
import librosa.display
import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences

base_dir = 'C:\\Users\\medep\\Downloads\\project dataset\\data'

audio_data = []
labels = []
dataset = []


max_length = 44100  


for label_dir in glob.iglob(os.path.join(base_dir, '*')):
    label = os.path.basename(label_dir)
    print(f'Processing label: {label}')
    
    for audio_file in glob.iglob(os.path.join(label_dir, '*.wav')):

        y, sr = librosa.load(audio_file, sr=None)
        y = pad_sequences([y], maxlen=max_length, padding='post', truncating='post')[0]

        audio_data.append(y)
        labels.append(label)
        duration = librosa.get_duration(path=audio_file)
        filename = os.path.basename(audio_file)
        if duration>= 3:
            slice_size = 3
            iterations = int((duration-slice_size)/(slice_size-1))
            iterations += 1
            initial_offset = (duration - ((iterations*(slice_size-1))+1))/2
            for i in range(iterations):
                offset = initial_offset + i*(slice_size-1)
                dataset.append({"filename": audio_file, "label": label, "offset":offset})
            
audio_data = np.array(audio_data)
labels = np.array(labels)

print("Data loading and preprocessing complete.")
print("Shape of audio data array:", audio_data.shape)
print("Shape of labels array:", labels.shape)

dataset = pd.DataFrame(dataset)
