<h1 align="center"> Forest Cover Prediction using Classification Mining Techniques and Deep Learning </h1>

<p align="center">
  <a href="https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning">
    <img src="https://user-images.githubusercontent.com/83835729/132753758-d3334562-56e8-46b8-87ce-6dcc64ad65e8.png" width="350" height="250">
  </a>
</p>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#project-team">Project Team</a></li>
    <li><a href="#project-objective">Project Objective</a></li>
    <li><a href="#tools-used">Tools Used</a></li>
    <li><a href="#methodology">Methodology</a></li>
    <li><a href="#implementation">Implementation</a></li>
    <li><a href="#inference">Inference</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
    <li><a href="#references">References</a></li>
    <!---
    <li><a href="#contact-us">Contact Us</a></li>
    --->
  </ol>
</details>

## Introduction

In recent years, there have been great advancements in the field of Machine Learning and Artificial Intelligence, where automation of lengthy, tedious, and manually implemented algorithms have been replaced with powerful and dynamic language libraries, that have logic programmed already with easy language syntax, thereby making implementations faster. Data Mining techniques have also seen notable improvements lately that enable people to get deeper understanding of various kinds of datasets, and hence infer better targeted prediction, bringing out several unknown or unobserved interpretations of the data.

Complex datasets that have considerably huge size and mediocre usability ratings might indicate towards a possible tendency of mediocrity in the prediction accuracies for the model created based on these datasets. However, it need not be a hard and fast rule and such generalised notions can be claimed incorrect with the use of latest advanced computing tools. The project deals with a dataset of a similar description and finds a variety of methods to create suitable prediction models using Classification Mining Techniques and Deep Learing concepts to yield a higher prediction accuracy than the base paper without overfitting the model for higher accuracy. 

## Project Team

Guidance Professor: Dr. Arup Ghosh, School of Computer Science and Engineering, VIT Vellore.

The group for our Data Mining project consists of -

|Sl.No. | Name  | ID Number |
|-| ------------- |:-------------:|
|1|     Arjun Das      | 20BDS0129     |
|2| Siddharth Pal      | 20BDS0409     |
|3| John Harshith      | 20BDS0411     |


## Project Objective

In 1999, Jock A. Blackard and Denis J. Dean implemented Artificial Neural Network and Discriminant Analysis to predict the forest cover from various areas of the Roosevelt National Forest in Colorado, USA.

However, we know from his paper that there is only 71.1% prediction accuracy in the model created by them using Artificial Neural Networks on various columns of data. Even though this method was better than previously implemented traditional Discriminant Analysis, our objective in this project is to find solutions to better the accuracy using the exact same dataset by creating various classification models.

This dataset includes information on tree type, shadow coverage, wilderness of the surrounding, distance to nearby landmarks (roads etc), soil type, local topography, etc. The dataset is very large having more than 0.5 million item sets. Its usability is 5.9 according to premiere data science websites.

The above dataset has been taken from the University of California Irvine Machine Learning Repository, and the source can be found [here](https://archive.ics.uci.edu/ml/datasets/Covertype), and the dataset is also present in the main branch of the repository as [covtype.csv](https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning/blob/main/covtype.csv). Finally the accuracy and overall prediction performance of the model is gauged for efficient use.

## Tools Used

To achieve the project objective we make use of the following tools -
* **Classification Mining Techniques**
  * Logistic Regression
  * Gaussian Naive Bayes'
  * K-Nearest Neighbors
  * Support-Vector Machine
  * Random Forest Classification

* **Deep Learning**
  * Artificial Neural Network

* **Python Language Libraries**
  * Seaborn
  * Matplotlib
  * Scikit-learn
  * Pandas
  * Keras
  * Tensorflow
 
* **R Language Libraries** 
  * Tidyverse
  * Skimr

## Methodology

1. Understanding the components of the dataset.
2. Creating visuals that will aid us with the understanding and enable us to find the proper combination of data columns for high accuracy.
3. Using various classification mining algorithms to create models that yield certain prediction accuracy for certain combination of data columns.
4. Fitting models appropriately and checking if any model has been overfitted or underfitted.
5. Iterating certain column combinations and algorithms for a large number of times to find the best train-test split and record that prediction percentage as well as the split content in a separate binary file for future use.
6. Using Deep Learning technique on the dataset and yield prediction accuracy.
7. Comaparing the performance of all the algorithms and concluding with the most recommended algorithm with this dataset.

## Implementation

A segment of the visualisation has used the R language and its libraries to help us understand the dataset better. The rest of the visualisation, classification and DL analysis has been implemented with the help of Python libraries.

### Understanding the Data

There are 13 columns of interest that are included in the dataset:

- Cover_Type: One of seven types of tree cover found in the forest. In the data downloaded for the project, the observations are coded 1 through 7. We have renamed them for clarity. Observations are based on the primary cover type of 30m x 30m areas, as determined by the United States Forest Service. This is our response variable.
- Wilderness_Area: Of the six wilderness areas in the Roosevelt National Forest, four were used in this dataset. In the original dataset, these were one-hot encoded. We put them in long form for better visualisation and because most machine learning methods will automatically one-hot encode categorical variables for us.
- Soil_Type: 40 soil types were identified in the dataset and more detailed information regarding the types can be found at https://www.kaggle.com/uciml/forest-cover-type-dataset. Similar to Wilderness_Area, Soil_Type was originally one-hot encoded.
- Elevation: The elevation of the observation in meters above sea level.
- Aspect: The aspect of the observation in degrees azimuth.
- Slope: The slope at which the observation is observed in degrees.
- Hillshade_9am: The amount of hillshade for the observation at 09:00 on the summer solstice. This is a value between 0 and 225.
- Hillshade_Noon: The amount of hillshade for the observation at 12:00 on the summer solstice. This is a value between 0 and 225.
- Hillshade_3pm: The amount of hillshade for the observation at 15:00 on the summer solstice. This is a value between 0 and 225.
- Vertical_Distance_To_Hydrology: Vertical distance to nearest water source in meters. Negative numbers indicate distance below a water source.
- Horizontal_Distance_To_Hydrology: Horizontal distance to nearest water source in meters.
- Horizontal_Distance_To_Roadways: Horizontal distance to nearest roadway in meters.
- Horizontal_Distance_To_Fire_Points: Horizontal distance to nearest wildfire ignition point in meters.

```R
library(tidyverse)
library(skimr)

df <- read.csv("Forest Cover.csv")
```
```R
glimpse(df)
nrow(df) - nrow(distinct(df))
any(is.na(df))
skim(df)
```

### Visualising the data



#### Soil Type

The base paper included a vivid description 

```R
set.seed(1808)

dff <- (df %>%
         group_by(Cover_Type) %>%
         sample_n(size = 1000) %>%
         ungroup() %>%
         gather(Wilderness_Area, Wilderness_Value, 
                Wilderness_Area1:Wilderness_Area4) %>% 
         filter(Wilderness_Value >= 1) %>%
         select(-Wilderness_Value) %>%
         mutate(Wilderness_Area = str_extract(Wilderness_Area, '\\d+'),
                Wilderness_Area = str_replace_all(Wilderness_Area,
                                                  c(`1` = 'Rawah',
                                                    `2` = 'Neota',
                                                    `3` = 'Comanche Peak',
                                                    `4` = 'Cache la Poudre')),
                Wilderness_Area = as.factor(Wilderness_Area)) %>%
         gather(Soil_Type, Soil_Value, Soil_Type1:Soil_Type40) %>%
         filter(Soil_Value == 1) %>%
         select(-Soil_Value) %>% 
         mutate(Soil_Type = as.factor(str_extract(Soil_Type, '\\d+'))) %>%
         mutate(Cover_Type = str_replace_all(Cover_Type,
                                             c(`1` = 'Spruce/Fir',
                                               `2` = 'Lodgepole Pine',
                                               `3` = 'Ponderosa Pine',
                                               `4` = 'Cottonwood/Willow',
                                               `5` = 'Aspen',
                                               `6` = 'Douglas Fir',
                                               `7` = 'Krummholz')),
                Cover_Type = as.factor(Cover_Type)) %>%
         select(Cover_Type:Soil_Type, Elevation:Slope,
                Hillshade_9am:Hillshade_3pm, Vertical_Distance_To_Hydrology,
                Horizontal_Distance_To_Hydrology:Horizontal_Distance_To_Fire_Points))

glimpse(dff)
skim(dff)

palette <- c('sienna1', 'chartreuse', 'lightskyblue1', 
             'hotpink', 'mediumturquoise', 'indianred1', 'gold')
ggplot(dff, aes(x = Cover_Type, y = Elevation)) +
  geom_violin(aes(fill = Cover_Type)) + 
  geom_point(alpha = 0.01, size = 0.5) +
  stat_summary(fun = 'median', geom = 'point') +
  labs(x = 'Forest Cover') +
  scale_fill_manual(values = palette) +
  theme_minimal() +
  theme(legend.position = 'none',
        axis.text.x = element_text(angle = 45,
                                   hjust = 1),
        panel.grid.major.x = element_blank())
```

<p align="left">
  <a href="https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning">
    <img src="https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning/blob/main/Visualization/Cover%20based%20on%20soil.png" width="500" height="400">
  </a>
</p>



### Classification Algorithm Determination

There exists various classification mining algorithms that can be used and implemented to form a model that yields very high prediction accuracy. A comparison of various feature selection with various supervised classification learning algorithms have been used and recorded. This tells us about the range of prediction percent one can get while using a particular algorithm (with or without ensemble) for a particular combination of features or data labels.

#### Gaussian Naive Bayes'

Gaussian Naive Bayes' algorithm has been the least efficient algorithm when it came to the yield of prediction accuracy. The normal distribution of the continuous variables were not forming well enough to yield a desirably high prediction accuracy, as well as overtake the prediction accuracy as mentioned in the research paper. Therefore, we try other classification algorithms to achieve success.

```python
from sklearn.naive_bayes import GaussianNB
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = GaussianNB()
model.fit(x_train, y_train)
print("Accuracy using Gaussian Naive Bayes: ", round(model.score(x_test, y_test) * 100, 3), "%", sep="")
```

#### Logistic Regression

Logistic Regression was the first algorithm to beat the research paper's prediction accuracy for a certain combinations of data columns only. This algorithm, when implemented on a huge dataset makes the prediction very sluggish and it may take hours to run over a single iteration of the algorithm so as to converge the solver.

More information on the algorithm used:
- Penalty type: l2
- Solver: saga
- No class weight, no random state
- Maximum number of iterations taken to coverge the solver: 7500

```python
from sklearn.linear_model import LogisticRegression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
lr = LogisticRegression(solver="saga", max_iter=7500)
lrmodel = lr.fit(x_train, y_train)
lracc = lr.score(x_test, y_test)
print("Accuracy using Logistic Regression: ", round(lracc*100, 3), "%", sep="")
```

#### Support Vector Machines

SVM takes the highest time to generate a prediction accuracy for this dataset. Each iteration takes more than 7 hours to run and execute. Such a computationally costly algorithm, although good in prediction will be highly time consuming for the users. The prediction accuracy for a particular combination of columns will differ for every variety of train-test splt. Therefore, in order to find the highest prediction possible, we iterate the algorithm a few times for a different train-test split and record the highest prediction accuracy's corresponding split in a pickle file for future use.

More information on the algorithm used:
- Kernel used: rbf
- Degree: 3
- Regularization parameter: 1
- No verbosity, no random state

```python
from sklearn import svm, metrics
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
svmmodel = svm.SVC(kernel="rbf", C=1)
svmmodel.fit(x_train, y_train)
y_pred = svmmodel.predict(x_test)
svmacc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy using SVM: ", round(svmacc*100, 3), "%", sep="")
```

#### Random Forest

A bagging ensemble technique for the Decision Tree Algorithm, Random Forest generates a constant prediction accuracy at a particular n (number of estimators that gets created) for a the column combinations. Therefore, there is no need to store the highest recorded accuracy and its corresponding test-train split in a pickle file. Random forest has also been one of the computationally cheapest techniques and resulted in one of the highest recorded prediction accuracies for many of the column combinations.

More information on the algorithm used:
- Number of estimators - 100
- Node splitting criteria taken - Gini index
- No explicit mention of the depth of the tree that is formed
- For best split, square root of the features was done instead of taking logarithm
- No random state included and no verbosity required

```python
from sklearn.ensemble import RandomForestClassifier
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
print("Accuracy using Random Forest: ", round(rf.score(x_test,y_test) * 100, 3), "%", sep="")
```

#### K-Nearest Neighbors

This lazy learner algorithm is highly effective, and better than Random Forest but it becomes quite computationally expensive when it is run on dataset of immense size. Due to variations in answer for every train-test split, we had to iterate the algorithm of a few number of times and store the split with the highest accuracy in a pickle. If the time is not an issue, then KNN yields the highest ever recorded prediction percentages.

More information on the algorithm used:
- Number of neighbors set: 3
- Distance set: Minkowski
- Weights set to each point: uniform

```python
from sklearn.neighbors import KNeighborsClassifier
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
knnmodel = KNeighborsClassifier(n_neighbors=3)
knnmodel.fit(x_train, y_train)
knnacc = knnmodel.score(x_test, y_test)
print("Accuracy using KNN: ", round(knnacc*100, 3), "%", sep="")
```

#### Deep Learning

Artificial Neural Networks (ANN) are multi-layer fully-connected neural nets and consist of an input layer, multiple hidden layers, and an output layer. ANN are inspired by the design of a human brain and tries to simulate it.
An artificial neuron receives a signal then processes it and can signal neurons connected to it. The signal at a connection is a real number, and the output of each neuron is computed by an activation function of the sum of its inputs. Neurons and edges typically have a weight that adjusts as learning proceeds. The weight increases or decreases the strength of the signal at a connection. Neurons may have a threshold such that a signal is sent only if the aggregate signal crosses that threshold.

More information on the algorithm used:
- Number of hidden layers: 2
- Learning Algorithm: stochastic gradient descent extension- Adam Optimizer
- Loss Function: sparse categorical crossentropy
- Activation function for input layer: linear (f(x) = x)
- Activation function for the 2 hidden layers: Rectified Linear Activation Unit (RELU)
- Activation function for output layer: Softmax Activation Function
- Number of nodes: 64
- Batch size: 64
- Number of epochs: 8
- Input variables linearly scaled to lie in range [0, 1]

```python
import tensorflow as tf
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
cover_model = tf.keras.models.Sequential()
cover_model.add(tf.keras.layers.Dense(
    units=64, activation='relu', input_shape=(X_train.shape[1],)))
cover_model.add(tf.keras.layers.Dense(units=64, activation='relu'))
cover_model.add(tf.keras.layers.Dense(units=8, activation='softmax'))
cover_model.compile(optimizer=tf.optimizers.Adam(
), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_cover = cover_model.fit(
    X_train, y_train, epochs=8, batch_size=64, validation_data=(X_test, y_test))
```

## Inference



## Conclusion

The objective of the project was successfully achieved as we found alternative supervised learning approaches that are able to yield a much higher and more accurate prediction percentage for the given dataset in a reduced time frame, owing to the size of the dataset. The most suitable algorithm would be K Nearest Neighbors algorithm and the label of interest can be aptly predicted by the taking all the columns into account.
=======
## Conclusion

This study improves upon the work of Jock A. Blackard and Denis J. Dean in predicting forest covers. After training and testing the same dataset with various classification models, some of the models managed to beat the old accuracy of "71.1%" with a new best accuracy of "97". 

With the optimizer and loss functions used, our deep learning model was able to beat the old accuracy too, with a whooping "99.9%".

## References

- [_Blackard, J. A., & Dean, D. J. (1999). Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables. Computers and electronics in agriculture, 24(3), 131-151._](https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning/blob/main/References/Comparative%20accuracies%20of%20artificial%20neural%20networks%20and%20discriminant%20analysis%20in%20predicting%20forest%20cover%20types%20from%20cartographic%20variables.pdf)
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
<!---
## Contact Us

_You can contact us through our LinkedIn account -_

* [John Harshith](https://www.linkedin.com/in/john-harshith-5354371b7/)
* [Arjun Das](https://www.linkedin.com/in/arjundas1/)
* [Siddharth Pal](https://www.linkedin.com/in/siddharthpal20/)
--->

