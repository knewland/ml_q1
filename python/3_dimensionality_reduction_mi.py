import numpy as np
import copy
from sklearn.feature_selection import mutual_info_classif as mi

'''
Dimensionality reduction using mutual information
Source: https://towardsdatascience.com/select-features-for-machine-learning-model-with-mutual-information-534fe387d5c8
https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif

Explanation: The mutual information score ranges from 0 to infinity. A low score indicates that the attribute has a weak correlation with the class, meaning it should be eliminated from the dataset that will be used to train the model.
'''



#reading file into array
with open('data_x.npy','rb') as f:
    data_x = np.load(f)
with open('data_y.npy','rb') as f:
    data_y = np.load(f)
tolerance = 0.2
mi_scores = mi(data_x, data_y)
print("Mutual information scores:")
print(mi_scores) #prints an array of all the mutual information scores for each attribute
#print(len(mi_scores))

mi_index = np.where(mi_scores > tolerance)[0]
data_x_new = data_x[:,mi_index]
print("Original number of attributes:", data_x.shape[1])
print("New number of attributes (MI):", data_x_new.shape[1])


#saving numpy array for next step
with open('mi_data_x.npy','wb') as f:
    np.save(f, data_x_new)
with open('mi_data_y.npy','wb') as f:
    np.save(f, data_y)
