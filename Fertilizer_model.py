import pandas as pd  # to read and manipulating data
import numpy as np  # to calculate mean and standard deviations
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# from sklearn.preprocessing import MinMaxScaler # to normalize data
from sklearn.preprocessing import LabelEncoder  # to encode object variable to numeric
from sklearn.model_selection import train_test_split  # to split data into trainin
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, classification_report
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

df = pd.read_csv("Fertilizer_Prediction.csv")

# Label Encoding
le = LabelEncoder()
df["Fertilizer"] = le.fit_transform(df["Fertilizer"])
df["Soil"] = le.fit_transform(df["Soil"])
df["Crop"] = le.fit_transform(df["Crop"])


X = df.drop(["Fertilizer"], axis=1)  # feature variables
y = df[["Fertilizer"]]  # Target variable

# Create train and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=10
)
result = {}


def plotter(x_model):
    plot_confusion_matrix(
        x_model,
        X_test,
        y_test,
        display_labels=[
            "DAP and MOP",
            "Good NPK",
            "MOP",
            "Urea and DAP",
            "Urea and MOP",
            "Urea",
            "DAP",
        ],
        xticks_rotation="vertical",
        cmap="YlGnBu",
    )


def score_report(y_pred_x, model_x):
    # print("Accuracy and report of model ",model_x)
    # print('Accuracy: ', accuracy_score(y_test, y_pred_x))
    result.update({model_x: accuracy_score(y_test, y_pred_x)})
    # print(classification_report(y_test, y_pred_x))


# to build a classification tree
model_dt = DecisionTreeClassifier(random_state=42)
model_dt = model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)

# plotter(model_dt)
score_report(y_pred_dt, model_dt)

# Random Forest model
model_RF = RandomForestClassifier(random_state=42)
model_RF = model_RF.fit(X_train, y_train)
y_pred_RF = model_RF.predict(X_test)

# plotter(model_RF)
score_report(y_pred_RF, model_RF)

# Gradient Boosting model
model_GB = GradientBoostingClassifier()
model_GB = model_GB.fit(X_train, y_train)
y_pred_GB = model_GB.predict(X_test)

# plotter(model_GB)
score_report(y_pred_GB, model_GB)

# Create K-Nearest Neighbors Classifier
model_knn = KNeighborsClassifier(n_neighbors=3)
model_knn = model_knn.fit(X_train, y_train)
y_pred_knn = model_knn.predict(X_test)

# plotter(model_knn)
score_report(y_pred_knn, model_knn)

# using SVM
model_svm = svm.SVC(kernel="linear")
model_svm.fit(X_train, y_train)
y_pred_svm = model_svm.predict(X_test)

# plotter(model_svm)
score_report(y_pred_svm, model_svm)

# using naive based
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
y_pred_nb = model_nb.predict(X_test)

# plotter(model_nb)
score_report(y_pred_nb, model_nb)

# reading the input from the file
lst = []
with open("input.txt", "r") as f:
    for i in f:
        data = i
        lst = list(data.split(","))

f.close()

# converting the input list to dataframe rwo
frame = pd.DataFrame([lst])

model = max(result, key=result.get)

y_pred = model.predict(frame)

if y_pred == 0:
    y_pred = "DAP"
elif y_pred == 1:
    y_pred = "DAP & MOP"
elif y_pred == 2:
    y_pred = "Good NPK"
elif y_pred == 3:
    y_pred = "MOP"
elif y_pred == 4:
    y_pred = "Urea"
elif y_pred == 5:
    y_pred = "Urea & DAP"
else:
    y_pred = "Urea & MOP"


# # writting the output in file
f_out = open("output.txt", "w")

f_out.write(y_pred)
f_out.close()

# print(y_pred)
