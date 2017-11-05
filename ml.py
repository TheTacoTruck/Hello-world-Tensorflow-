import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sklearn
from sklearn import model_selection
from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'petal-length', 'sepal-width', 'peta;-width', 'class']
dataset = pandas.read_csv(url, names=names)

# print(dataset.describe()) --- (summary)
# print(dataset.shape)  ---- (dimensions)
# print(dataset.head(20)) --- (peek)
# print(dataset.groupby('class').size()) --- (class distribution)

# data visualization
# two types of plots
# 1. Univariate plots to better understand each attribute
# 2. Multivariate plots to better understand the relationships between attributes

# dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
# plt.show() --- Box and whisker

# dataset.hist()
# plt.show() --- Histogram (Gaussian distribution)

# scatter_matrix(dataset)
# plt.show() --- Multivariate (Scatter / histogram)


# Create a validation dataset
# split the dataset in two 80% for training. 20% for validation
array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)
# Test Harness
# Ten fold cross to estimate accuracy
# split training set in 10. Train on 9, test on 1 repeat for all variations

# Test options and evaluate metric
seed = 7
scoring = 'accuracy'

# Build models
# Evaluate six different models
#   Logical Regression (LR)
#   Linear Discriminant analysis (LDA)
#   K-Nearest Neighbors (KNN)
#   Classification and Regression Trees (CART)
#   Gaussian Naive Bayes
#   Support Vector Machines (SVM)

# Spot check algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', (DecisionTreeClassifier())))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Compare algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
# plt.show()

# make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
