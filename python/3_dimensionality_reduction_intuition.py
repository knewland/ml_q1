import numpy as np
import copy

with open('data_x.npy','rb') as f:
    data_x = np.load(f)
with open('data_y.npy','rb') as f:
    data_y = np.load(f)


#summing nb_[special character] attribute values
nb_combined=np.sum(data_x[:,4:23],axis=1)
nb_combined = np.vstack(nb_combined)

#new array with the summed value
new_data_x = np.concatenate((data_x, nb_combined),1)

num_instances = new_data_x.shape[0] #total number of instances

#searching for binary attributes that have low variance
oneCount = 0
zeroCount = 0
num = 0
binaryCount = 0
tolerance = 0.02 #if a binary attribute has a variance of less than 2%, then it will be deleted

#automatically will delete colums 4-22 as they have been summed
deletion = set([])
for i in range(4,23):
    deletion.add(i)
for column in range(new_data_x.shape[1]):
    for i in new_data_x[:,column]:
        if i == 1:
            oneCount += 1;
        elif i == 0:
            zeroCount += 1
    if oneCount + zeroCount == num_instances:
        if((float(oneCount)/float(num_instances) < tolerance) or (float(zeroCount)/float(num_instances) < tolerance)):
            deletion.add(num)
        binaryCount += 1
    zeroCount = 0
    oneCount = 0
    num += 1

#deleting columns
dlt = list(deletion)
new_data_x = np.delete(new_data_x, dlt, axis=1)

print()
print("new number of attributes: ")
print(new_data_x.shape[1])

#saving numpy array for next step
with open('intuition_data_x.npy','wb') as f:
    np.save(f, new_data_x)
with open('intuition_data_y.npy','wb') as f:
    np.save(f, data_y)


