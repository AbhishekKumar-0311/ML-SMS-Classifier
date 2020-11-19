# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## SMS Spam Classifier: Multinomial Naive Bayes
#
# The notebook is divided into the following sections:
# 1. Importing and preprocessing data
# 2. Building the model: Multinomial Naive Bayes
#     - Model building 
#     - Model evaluation

# ### 1. Importing and Preprocessing Data

# +
import pandas as pd

# reading the training data
docs = pd.read_table('SMSSpamCollection', header=None, names=['Class', 'sms'])
docs.head()
# -

# number of SMSes / documents
len(docs)

# counting spam and ham instances
ham_spam = docs.Class.value_counts()
ham_spam

print("spam rate is about {0}%".format(
    round((ham_spam[1]/float(ham_spam[0]+ham_spam[1]))*100), 2))

# mapping labels to 0 and 1
docs['label'] = docs.Class.map({'ham':0, 'spam':1})

docs.head()

# we can now drop the column 'Class'
docs = docs.drop('Class', axis=1)
docs.head()

# convert to X and y
X = docs.sms
y = docs.label
print(X.shape)
print(y.shape)

# splitting into test and train
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

X_train.head()

y_train.head()

# vectorizing the sentences; removing stop words
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english')

vect.fit(X_train)

# printing the vocabulary
vect.vocabulary_

# vocab size
len(vect.vocabulary_.keys())

# transforming the train and test datasets
X_train_transformed = vect.transform(X_train)
X_test_transformed = vect.transform(X_test)

# note that the type is transformed (sparse) matrix
print(type(X_train_transformed))
print(X_train_transformed)

# ### 2. Building and Evaluating the Model

# +
# training the NB model and making predictions
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

# fit
mnb.fit(X_train_transformed,y_train)

# predict class
y_pred_class = mnb.predict(X_test_transformed)

# predict probabilities
y_pred_proba = mnb.predict_proba(X_test_transformed)

# -

# note that alpha=1 is used by default for smoothing
mnb

# ### Model Evaluation

# printing the overall accuracy
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)

# confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)
# help(metrics.confusion_matrix)

confusion = metrics.confusion_matrix(y_test, y_pred_class)
print(confusion)
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]
TP = confusion[1, 1]

sensitivity = TP / float(FN + TP)
print("sensitivity",sensitivity)

specificity = TN / float(TN + FP)
print("specificity",specificity)

# **_The 0.99 value of specificity ensures that most of the true negative i.e, Non-Spam messages are categorised correctly which reduces the chance of missing the genuine messages._**

precision = TP / float(TP + FP)
print("precision",precision)
print(metrics.precision_score(y_test, y_pred_class))

print("precision",precision)
print("PRECISION SCORE :",metrics.precision_score(y_test, y_pred_class))
print("RECALL SCORE :", metrics.recall_score(y_test, y_pred_class))
print("F1 SCORE :",metrics.f1_score(y_test, y_pred_class))

y_pred_class

y_pred_proba

# +
# creating an ROC curve
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred_proba[:,1])
roc_auc = auc(false_positive_rate, true_positive_rate)
# -

# area under the curve
print (roc_auc)

# matrix of thresholds, tpr, fpr
pd.DataFrame({'Threshold': thresholds, 
              'TPR': true_positive_rate, 
              'FPR':false_positive_rate
             })

# plotting the ROC curve
# %matplotlib inline  
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.plot(false_positive_rate, true_positive_rate)

# ### Conclusion
#
# **_We can conclude that the model built performs well with a close to ideal ROC curve, also confirmed by .99 roc-auc value._**


