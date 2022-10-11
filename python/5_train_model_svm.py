import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn import metrics
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# Split dataset into random train and test subsets:
prefixes = ['intuition', 'pca', 'mi', 'us', 'tb']
for i in range(len(prefixes)):
    filename = prefixes[i] + '_xtrain.csv'
    file = open(filename)
    train_x = np.genfromtxt(file, delimiter=',')
    file.close()

    filename = prefixes[i] + '_xtest.csv'
    file = open(filename)
    test_x = np.genfromtxt(file, delimiter=',')
    file.close()
    
    filename = prefixes[i] + '_ytrain.csv'
    file = open(filename)
    train_y = np.genfromtxt(file, delimiter=',')
    file.close()
    
    filename = prefixes[i] + '_ytest.csv'
    file = open(filename)
    test_y = np.genfromtxt(file, delimiter=',')
    file.close()
    
    sc = StandardScaler()
    sc.fit(train_x)
    x_train_std = sc.transform(train_x)
    x_test_std = sc.transform(test_x)
    
    svm = SVC(kernel='linear', random_state=1, C=0.1)
    svm.fit(x_train_std, train_y)

    y_pred = svm.predict(x_test_std)

    acc_score = accuracy_score(test_y, y_pred)
    auc_score = roc_auc_score(test_y, y_pred)
    #print(acc_score)
    
    conf_mat = confusion_matrix(test_y, y_pred, labels = [0,1])
    
    #print()
    print(prefixes[i], "SVM Confusion Matrix")
    print(conf_mat)
    trueNeg, falsePos, falseNeg, truePos = conf_mat.ravel()
    fpr, tpr, threshold = metrics.roc_curve(test_y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print("True Phishing: ", trueNeg)
    print("True Legitimate: ", truePos)
    print("False Phishing: ", falseNeg)
    print("False Legitimate: ", falsePos)
    rate = (trueNeg * 1.0) / ((falsePos + trueNeg) * 1.0)
    print(str(prefixes[i]) + " Overall SVM Accuracy: " + str(acc_score))
    print("AUC score: ", auc_score)
    print("Phishing link CORRECT classification rate: ", rate)
    print()
    print()
    
    plt.title(str(prefixes[i]) + ' SVM Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.5f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(str(prefixes[i]) + "_svm_roc.png")
    plt.clf()

    #plt.show()