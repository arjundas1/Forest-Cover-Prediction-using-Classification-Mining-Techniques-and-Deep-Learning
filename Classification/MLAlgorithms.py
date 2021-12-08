import pandas as pd
import sklearn
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing, svm, metrics

df = pd.read_csv("Forest Cover.csv")
scale = StandardScaler()
le = preprocessing.LabelEncoder()
cov = le.fit_transform(list(df["Cover_Type"]))
y = list(cov)
'''
x = df[["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm",
        "Horizontal_Distance_To_Fire_Points", "Wilderness_Area1", "Wilderness_Area2",
        "Wilderness_Area3", "Wilderness_Area4"]]
'''
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

model = GaussianNB()
model.fit(x_train, y_train)
print("Accuracy using Gaussian Naive Bayes: ", round(model.score(x_test, y_test) * 100, 3), "%", sep="")

knnmodel = KNeighborsClassifier(n_neighbors=3)
knnmodel.fit(x_train, y_train)
knnacc = knnmodel.score(x_test, y_test)
print("Accuracy using KNN: ", round(knnacc*100, 3), "%", sep="")

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
print("Accuracy using Random Forest: ", round(rf.score(x_test,y_test) * 100, 3), "%", sep="")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
lr = LogisticRegression(solver="saga", max_iter=7500)
lrmodel = lr.fit(x_train, y_train)
lracc = lr.score(x_test, y_test)
print("Accuracy using Logistic Regression: ", round(lracc*100, 3), "%", sep="")

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
svmmodel = svm.SVC(kernel="rbf", C=1)
svmmodel.fit(x_train, y_train)
y_pred = svmmodel.predict(x_test)
svmacc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy using SVM: ", round(svmacc*100, 3), "%", sep="")
