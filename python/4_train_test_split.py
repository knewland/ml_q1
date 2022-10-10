import numpy as np
from sklearn.model_selection import train_test_split

prefixes = ['intuition', 'pca', 'mi', 'us', 'tb']

for i in range(len(prefixes)):
    with open(prefixes[i]+'_data_x.npy','rb') as f:
        data_x = np.load(f)
    with open(prefixes[i]+'_data_y.npy','rb') as f:
        data_y = np.load(f)

    #split into train test set: simple random sampling
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25);

    y_train = np.vstack(y_train)
    train = np.concatenate((x_train, y_train),1)

    y_test = np.vstack(y_test)
    test = np.concatenate((x_test, y_test),1)

    np.savetxt(prefixes[i]+"_xtrain.csv", x_train, fmt="%f", delimiter=",")
    np.savetxt(prefixes[i]+"_xtest.csv", x_test, fmt="%f", delimiter=",")
    np.savetxt(prefixes[i]+"_ytrain.csv", y_train, fmt="%f", delimiter=",")
    np.savetxt(prefixes[i]+"_ytest.csv", y_test, fmt="%f", delimiter=",")

print("Number of  training instances:", len(train))
print("Number of  testing instances",len(test))
print()