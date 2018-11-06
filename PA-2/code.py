
#import pandas as pd
import pickle
import numpy as np
from sklearn.naive_bayes import BernoulliNB 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



print("Loading datasets...")
Xs = pickle.load(open('binarized_xs.pkl', 'rb'))
ys = pickle.load(open('binarized_ys.pkl', 'rb'))
print("Done.")

train_jll = np.zeros((10, 15))

# Accuracies on the test set.
test_jll = np.zeros((10, 15))

## todo 1
prior= [10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7]

for i in range(0,10):
    x_train, x_test, y_train, y_test = train_test_split(Xs[i], ys[i], test_size=1./3, random_state=3330)

    class_size_train=len(y_train)
    
    class_size_test=len(y_test)
    
    for j in range(0,15):
        
        clf = BernoulliNB(alpha=prior[j], binarize=0.0 , class_prior=None , fit_prior=True  )

        clf = clf.fit(x_train, y_train)
        
        a= clf._joint_log_likelihood(x_train)

        # calculating joint log likelyhood of train dataset
        
        sum1=0
        for k in range(0,class_size_train):
            if y_train[k].__eq__(False):
                sum1 += a[k][0]
            else:
                sum1 += a[k][1]

        train_jll[i][j] = sum1

        a= clf._joint_log_likelihood(x_test)

        # calculating joint log likelyhood of test dataset
        
        sum2=0
        for k in range(0,class_size_test):
            if y_test[k].__eq__(False):
                sum2 += a[k][0]
            else:
                sum2 += a[k][1]


        test_jll[i][j] = sum2
            
## DO NOT MODIFY BELOW THIS LINE.

print(" joint log likelyhood Train set ")
for i in range(10):
	print("\t".join("{0:.4f}".format(n) for n in train_jll[i]))
	

print("\n joint log likelyhood Test set")
for i in range(10):
	print("\t".join("{0:.4f}".format(n) for n in test_jll[i]))

 
pickle.dump((train_jll, test_jll), open('result.pkl', 'wb'))