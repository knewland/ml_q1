import numpy as np
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB # or any other NB model
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
    
    #https://stackoverflow.com/questions/34725726/is-it-possible-apply-pca-on-any-text-classification?noredirect=1&lq=1, so normalize PCA attributes to between 0 and 1 back in dimensionality reduction
    nb_classifier = MultinomialNB()
    nb_classifier.fit(train_x, train_y)

    y_pred = nb_classifier.predict(test_x)

    acc_score = accuracy_score(test_y, y_pred)
    auc_score = roc_auc_score(test_y, y_pred)
    #print(acc_score)
    
    conf_mat = confusion_matrix(test_y, y_pred, labels = [0,1])
    
    #print()
    print(prefixes[i], "Naive Bayes Confusion Matrix")
    print(conf_mat)
    trueNeg, falsePos, falseNeg, truePos = conf_mat.ravel()
    fpr, tpr, threshold = metrics.roc_curve(test_y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    print("True Phishing: ", trueNeg)
    print("True Legitimate: ", truePos)
    print("False Phishing: ", falseNeg)
    print("False Legitimate: ", falsePos)
    rate = (trueNeg * 1.0) / ((falsePos + trueNeg) * 1.0)
    print(str(prefixes[i]) + " Overall Naive Bayes Accuracy: " + str(acc_score))
    print("AUC score: ", auc_score)
    print("Phishing link CORRECT classification rate: ", rate)
    print()
    print()
    
    plt.title(str(prefixes[i]) + ' Naive Bayes Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.5f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(str(prefixes[i]) + "_nb_roc.png")
    plt.clf()

    #plt.show()