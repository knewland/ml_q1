import numpy as np
import copy
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2

'''
Dimensionality reduction using Univariate Selection
Source: https://scikit-learn.org/stable/modules/feature_selection.html
'''

#reading file into array
with open('data_x.npy','rb') as f:
    data_x = np.load(f)
with open('data_y.npy','rb') as f:
    data_y = np.load(f)

#selects top 10 percent of attributes
x_new = SelectPercentile(chi2, percentile=10).fit_transform(data_x, data_y)
print(x_new.shape)

#saving numpy array for next step
with open('us_data_x.npy','wb') as f:
    np.save(f, x_new)
with open('us_data_y.npy','wb') as f:
    np.save(f, data_y)