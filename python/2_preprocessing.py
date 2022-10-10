import numpy as np
import copy

#reading dataset into array
filename = 'dataset_phishing_classes_renamed.csv'
file = open(filename)
file.readline() #reads first line from dataset to not include in array
data = np.genfromtxt(file, delimiter=',')
file.close()

#reading data into attributes and classes
data_x, data_y = data[:, 0:-1], data[:, -1]

#getting number of attributes for calculating mean
print("original number of attributes:")
print(data_x.shape[1])

#replacing invalid domain_age values
data_x_copy = copy.deepcopy(data_x)
domain_age_index = 82
instances = data_x_copy.shape[0] #total number of instances in dataset
print("instances", instances)

#determining the mean, so replacing invalid values (-1) with 0
data_x_copy[data_x_copy<0] = 0
print(data_x_copy[:,domain_age_index])

#summing the values under the domain_age attribute
domain_age_sum = np.sum(data_x_copy[:,domain_age_index],axis=0)

#calculating mean using sum and number of instances
mean = float(domain_age_sum) / float(instances)
print("mean is",mean)
print("domain age total is", domain_age_sum)

#overwriting the invalid values with the mean
data_x[data_x<0] = mean
print()
print(data_x[:,domain_age_index])

#saving numpy array for next step
with open('data_x.npy','wb') as f:
    np.save(f, data_x)
with open('data_y.npy','wb') as f:
    np.save(f, data_y)


