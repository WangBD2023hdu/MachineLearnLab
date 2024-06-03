import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

# my-self implementation model

import myselfmodel as MyModel

# Get the data
df = pd.read_excel(os.path.join('data', 'Dry_Bean_Dataset.xlsx'))
df.duplicated().sum()
df.drop_duplicates(inplace=True)

df.isna().sum(axis=1).sum() # is 0
# see the dataset in the lab directory
# df = pd.read_excel(os.path.join('data', 'Dry_Bean_Dataset.xlsx'), usecols=['Area', 'Perimeter'])
data = df.drop(columns=['Area', 'Class'])
fig, axis = plt.subplots(int((len(df.columns)+1) / 2), 2, figsize=(24, 30))
for i, col in enumerate(df.columns):
    axis[i//2, i % 2].hist(df[col])
    axis[i//2, i % 2].set_title(col)
plt.savefig("distribution.png")
plt.show()

plt.figure(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True)
plt.savefig("hot_2.png")

# to show decision tree clearly
data = data[['MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'ConvexArea', 'EquivDiameter']]
label = df[['Class']]
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(train_data, train_label)
y_pred = clf.predict(test_data)
accuracy = accuracy_score(test_label, y_pred)


print(f'Accuracy: {accuracy:.2f}')
print("\nClassification Report:")
print(classification_report(test_label, y_pred))
report = classification_report(test_label, y_pred)
print("\nConfusion Matrix:")
print(confusion_matrix(test_label, y_pred))
# myself model

dtmodel = MyModel.DecisionTreeClassifier(10)
label_ = [i[0] for i in train_label.values.tolist()]
dtmodel.fit(train_data.values.tolist(), label_)
predict_y = dtmodel.predict(test_data.values.tolist())

accuracy = accuracy_score(test_label, predict_y)

print(f'Accuracy: {accuracy:.2f}')
print("\nClassification Report:")
print(classification_report(test_label, predict_y))
report = classification_report(test_label, predict_y)
print("\nConfusion Matrix:")
print(confusion_matrix(test_label, predict_y))

# plt.figure(figsize=(20, 12), dpi=1200)
# plot_tree(clf, filled=True, feature_names=list(train_data.columns), class_names=list(set(train_label['Class'])))
# plt.savefig("tree.png")

## KNN

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

acc = 0
max_acc_k = 12

mclass_report = None
mconf_matrix = None
# 选择K值
for k in range(12, 20):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_label)
    y_pred = knn.predict(test_data)
    conf_matrix = confusion_matrix(test_label, y_pred)
    class_report = classification_report(test_label, y_pred)
    accuracy = accuracy_score(test_label, y_pred)
    if accuracy > acc:
        acc = accuracy
        max_acc_k = k
        mclass_report = class_report
        mconf_matrix = conf_matrix
print("Confusion Matrix:\n", mconf_matrix)
print("\nClassification Report:\n", mclass_report)
print("\nAccuracy:", max_acc_k)




acc = 0
max_acc_k = 12

mclass_report = None
mconf_matrix = None
# 选择K值
label = [lab[0] for lab in train_label.values.tolist()]
for k in range(12, 20):
    KN = MyModel.KNearestNeighbors(k=k)
    KN.fit(train_data.tolist(), label)
    y_pred = KN.predict(test_data)
    conf_matrix = confusion_matrix(test_label, y_pred)
    class_report = classification_report(test_label, y_pred)
    accuracy = accuracy_score(test_label, y_pred)
    if accuracy > acc:
        acc = accuracy
        max_acc_k = k
        mclass_report = class_report
        mconf_matrix = conf_matrix
print("Confusion Matrix:\n", mconf_matrix)
print("\nClassification Report:\n", mclass_report)
print("\nAccuracy:", max_acc_k)




