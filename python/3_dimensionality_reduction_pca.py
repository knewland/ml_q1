import numpy as np
import copy
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

'''
Dimensionality reduction using PCA
Source: https://machinelearningmastery.com/feature-selection-machine-learning-python/
https://www.mikulskibartosz.name/pca-how-to-choose-the-number-of-components/

****http://www.billconnelly.net/?p=697****** redo with this

Explanation:
'''

#reading file into array
with open('data_x.npy','rb') as f:
    data_x = np.load(f)
with open('data_y.npy','rb') as f:
    data_y = np.load(f)

    
#Normalizing the x-axis data between 0 and 1 before PCA is performed
scaler = MinMaxScaler()
rescaled_data_x = scaler.fit_transform(data_x)

#PCA
explained_variance = 0.95
pca = PCA(n_components=explained_variance)
pca.fit(rescaled_data_x)
reduced = pca.transform(rescaled_data_x)

#this is new
scaler = MinMaxScaler()
reduced = scaler.fit_transform(reduced)
print(reduced.shape)



#saving numpy array for next step
with open('pca_data_x.npy','wb') as f:
    np.save(f, reduced)
with open('pca_data_y.npy','wb') as f:
    np.save(f, data_y)
