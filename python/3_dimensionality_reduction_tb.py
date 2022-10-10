import numpy as np
import copy
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

'''
Dimensionality reduction using Tree-based feature selection
Source: https://scikit-learn.org/stable/modules/feature_selection.html
'''

#reading file into array
with open('data_x.npy','rb') as f:
    data_x = np.load(f)
with open('data_y.npy','rb') as f:
    data_y = np.load(f)
print(data_x.shape)
print(data_y.shape)

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(data_x, data_y)

print("feature importances:")
print(clf.feature_importances_)

model = SelectFromModel(clf, prefit=True)
data_x_new = model.transform(data_x)
print(data_x_new.shape)
#saving numpy array for next step
with open('tb_data_x.npy','wb') as f:
    np.save(f, data_x_new)
with open('tb_data_y.npy','wb') as f:
    np.save(f, data_y)
