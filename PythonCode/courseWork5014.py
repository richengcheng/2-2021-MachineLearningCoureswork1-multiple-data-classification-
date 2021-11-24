
import numpy as numpy
import pandas as panda
from pandas import DataFrame
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve



#add header topic names to the dataset
data_Head_names = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
dataSet = panda.read_csv('./data/crx.data', sep=',', header=None, names = data_Head_names)

#replace the '?' with nan value in the dataset and delete rows that contain it
dataSet.replace('?', numpy.nan, inplace=True)
dataSet.dropna(axis = 0, inplace=True)

#Categorical variables be encoded using dummy variables
dataSet = panda.get_dummies(dataSet, columns=['A1', 'A4', 'A5', 'A6', 'A7', 'A9', 'A10', 'A12', 'A13'], drop_first=True)


# set lables for feature of data
mapping_output = {
    '+': 1,
    '-': 0
}

dataSet['A16'] = dataSet['A16'].map(mapping_output)
#transfer the type of data in A2 A14 to float
dataSet['A2'] = panda.to_numeric(dataSet['A2'], errors='coerce')
dataSet['A14'] = panda.to_numeric(dataSet['A14'], errors='coerce')

#split the dataset into inputdata and outputdata
data_Input = dataSet.loc[:, dataSet.columns != 'A16']
data_output = dataSet['A16']

#split the dataset into trainset and testset
trainData_X, testData_X, trainData_y, testData_y = train_test_split(data_Input, data_output, test_size=0.25, stratify= data_output, random_state=1)


def Outliers(x):
    a = trainData_X[x].quantile(0.75)
    b = trainData_X[x].quantile(0.25)
    c = trainData_X[x]
    c.values[(c >= (a - b) * 1.5 + a) | (trainData_X[x] <= b - (a - b) * 1.5)] = c.median()



Outliers('A2')
Outliers('A3')
Outliers('A8')
Outliers('A11')
Outliers('A14')
Outliers('A15')

#Scalring train Data
min_max_scalre = preprocessing.MinMaxScaler()
New_train_X = min_max_scalre.fit_transform(trainData_X)
New_train_X = DataFrame(New_train_X)

#Scalring Test Data
New_test_X = min_max_scalre.fit_transform(testData_X)
New_test_X = DataFrame(New_test_X)


#recursive filtering of data values
function_rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=2)
model = GradientBoostingClassifier()
pipe = Pipeline([('Feature Selection', function_rfe), ('Model', model)])
cv_Function = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)
n_scores = cross_val_score(pipe, New_train_X, trainData_y, scoring='accuracy', cv=cv_Function)
numpy.mean(n_scores)
pipe.fit(New_train_X, trainData_y)
print(panda.DataFrame(function_rfe.support_, index=New_train_X.columns, columns=['Rank']))

#reseting train Data
New_train_X = New_train_X[[4, 32]]
print(New_train_X)

#reseting test Data
New_test_X = New_test_X[[4, 32]]



#LogisticRegression
LogIsticG = LogisticRegression(penalty='none', class_weight='balanced')
LogIsticG.fit(New_train_X, trainData_y)
print(LogIsticG.coef_)
print(LogIsticG.intercept_)
y_prediction = LogIsticG.predict(New_test_X)


#evalution
print('evalution')
cm = confusion_matrix(testData_y, y_prediction)
print(cm)
scores = cross_val_score(estimator=LogIsticG, X=New_train_X, y=trainData_y, cv=10, n_jobs=1)
print(scores)
print('CV accuracy: %.3f +/- %.3f' % (numpy.mean(scores), numpy.std(scores)))


fontsize = "21";
params = {'figure.autolayout':True,
          'legend.fontsize': fontsize,
          'figure.figsize': (12, 8),
          'axes.labelsize': fontsize,
          'axes.titlesize': fontsize,
          'xtick.labelsize':9,
          'ytick.labelsize':9 }
plt.rcParams.update(params)

plt.figure(1)
plt.title('Precision/Recall Curve')# give  a title
plt.xlabel('Recall')# make labels
plt.ylabel('Precision')


#plot_precision_recall_curve
scoreForY = LogIsticG.predict_proba(New_test_X)[:, 1]
precision, recall, thresholds = precision_recall_curve(testData_y, scoreForY)
plt.figure(1)
plt.plot(precision, recall)
plt.show()

#average_precision_score
from sklearn.metrics import  average_precision_score
ap_score= average_precision_score(testData_y, scoreForY)
print(ap_score)

