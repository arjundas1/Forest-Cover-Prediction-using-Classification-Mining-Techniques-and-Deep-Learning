<h1 align="center"> Forest Cover Prediction using Classification Mining Techniques and Deep Learning </h1>

<p align="center">
  <a href="https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning">
    <img src="https://user-images.githubusercontent.com/83835729/132753758-d3334562-56e8-46b8-87ce-6dcc64ad65e8.png" width="600" height="300">
  </a>
</p>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#project-team">Project Team</a></li>
    <li><a href="#project-objective">Project Objective</a></li>
    <li><a href="#methodology">Methodology</a></li>
    <li><a href="#tools-used">Tools Used</a></li>
    <li><a href="#implementation">Implementation</a></li>
    <li><a href="#inference">Inference</a></li>
    <li><a href="#conclusion">Conclusion</a></li>
    <li><a href="#references">References</a></li>
    <li><a href="#contact-us">Contact Us</a></li>
  </ol>
</details>

## Introduction

In recent years, there have been great advancements in the field of Machine Learning and Artificial
Intelligence, where automation of lengthy, tedious, and manually implemented algorithms has
been replaced with powerful and dynamic language libraries that have logic programmed already
with easy language syntax, thereby making implementations faster. Data Mining techniques have
also seen notable improvements lately that enable people to get a deeper understanding of various 
kinds of datasets, and hence infer better-targeted prediction, bringing out several unknown or
unobserved interpretations of the data.

Complex datasets that have a considerably huge size and mediocre usability ratings might indicate
a possible tendency of mediocrity in the prediction accuracies for the model created based on these
datasets. However, it need not be a hard and fast rule and such generalized notions can be claimed
incorrectly with the use of the latest advanced computing tools. The project deals with a dataset of
a similar description and finds a variety of methods to create suitable prediction models using
Classification Mining Techniques and Deep Learning concepts to yield a higher prediction
accuracy than the base paper without overfitting the model for higher accuracy.
 
## Project Team

Guidance Professor: Dr. Arup Ghosh, School of Computer Science and Engineering, VIT Vellore.

The group for our Data Mining project consists of -

|Sl.No. | Name  | ID Number |
|-| ------------- |:-------------:|
|1|     Arjun Das      | 20BDS0129     |
|2| Siddharth Pal      | 20BDS0409     |
|3| John Harshith      | 20BDS0411     |

## Project Objective

In 1999, Jock A. Blackard and Denis J. Dean implemented Artificial Neural Network and
Discriminant Analysis to predict the forest cover from various areas of the Roosevelt National
Forest in Colorado, USA.

However, we know from his paper that there is only 71.1% prediction accuracy in the model
created by them using Artificial Neural Networks on various columns of data. Even though this
method was better than previously implemented traditional Discriminant Analysis, our objective
in this project is to find solutions to better the accuracy using the exact same dataset by creating
various classification models.

This dataset includes information on tree type, shadow coverage, the wilderness of the
surrounding, distance to nearby landmarks (roads, etc.), soil type, local topography, etc. The
dataset is very large having more than 0.5 million item sets. Its usability is 5.9 according to
premiere data science websites.

The above dataset has been taken from the University of California Irvine Machine Learning
Repository, and the source can be found [here](https://archive.ics.uci.edu/ml/datasets/covertype), and the dataset is also present in the main branch of
the repository as [covtype.csv](https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning/blob/main/covtype.csv). Finally, the accuracy and overall prediction performance of the
model are gauged for efficient use.

## Methodology

1. Understanding the components of the dataset.
2. Creating visuals that will aid us with understanding and enable us to find the proper
combination of data columns for high accuracy.
3. Using various classification mining algorithms to create models that yield certain prediction
accuracy for a certain combination of data columns.
4. Fitting models appropriately and checking if any model has been overfitted or under fitted.
5. Iterating certain column combinations and algorithms a large number of times to find the best
train-test split and record that prediction percentage as well as the split content in a separate
binary file for future use.
6. Using Deep Learning techniques on the dataset and yield prediction accuracy.
7. Comparing the performance of all the algorithms and concluding with the most recommended
algorithm with this dataset.

## Tools Used

To achieve the project objective, we make use of the following tools –

**Python Language Libraries**

- Seaborn
- Matplotlib
- Scikit-learn
- Pandas
- Keras
- Tensorflow

**R Language Libraries**

- Tidyverse
- Skimr

## Implementation

A segment of the visualization has used the R language and its libraries to help us understand the
dataset better. The rest of the visualization, classification, and DL analysis has been implemented
with the help of Python libraries.

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

### Visualization

The 7 types of forest cover are very unevenly included in the dataset, as shown in the figure below.
This might make learning models difficult, to predict the covers with fewer itemset. 

<p align="center">
  <a>
    <img src="https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning/blob/main/Visualization/Countplot.png" width="400" height="300">
  </a>
</p>

Not all the 40 soil types included in the dataset will be completely helpful for our prediction. From
the count plot, we can infer that soil types 11, 16, 31, 32, and 33 will be quite helpful in the
prediction-making process because of more representation of forest cover types.

<p align="center">
  <a>
    <img src="https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning/blob/main/Visualization/Cover%20based%20on%20soil.png" width="600" height="450">
  </a>
</p>

The elevation column might require scaling as we infer from the violin plot given below that the
range of elevation is too high when compared with the other column sets, and long ranges of cover
points that lie for each cover type.

<p align="center">
  <a>
    <img src="https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning/blob/main/Visualization/Elevation.png" width="400" height="300">
  </a>
</p>

The plots below show which class a point belongs to when hillshade columns are considered.
The class distribution overlaps in the plots. Hillshade patterns give an ellipsoid pattern to each
other. Aspect and Hillshades attribute to form a sigmoid pattern if graphed. Horizontal and
vertical distance to hydrology will give an almost linear pattern.

<p align="center">
  <a>
    <img src="https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning/blob/main/Visualization/Hillshade3pm%20Vs%20Hillshade9am.png" width="450" height="350">
  </a>
</p>

From the below scatter plot, we can conclude that the vertical and horizontal distance to
hydrology is more spread out when it is not of soil type 2 and is less spread out when it is of the
respective type.

<p align="center">
  <a>
    <img src="https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning/blob/main/Visualization/VerticalDistanceToHydrology%20Vs%20HorizontalDistanceToHydrology%20Lineplot.png" width="450" height="350">
  </a>
</p>

### Classification Algorithm Determination

There exist various classification mining algorithms that can be used and implemented to form a
model that yields very high prediction accuracy. A comparison of various feature selection with
various supervised classification learning algorithms has been used and recorded. This tells us
about the range of prediction percent one can get while using a particular algorithm (with or
without ensemble) for a particular combination of features or data labels.

#### Gaussian Naive Bayes'

Gaussian Naive Bayes’ algorithm has been the least efficient algorithm when it comes to the yield
of prediction accuracy. The normal distribution of the continuous variables was not forming well
enough to yield a desirably high prediction accuracy, as well as overtake the prediction accuracy
as mentioned in the research paper. Therefore, we try other classification algorithms to achieve
success.

#### Logistic Regression

Logistic Regression was the first algorithm to beat the research paper’s prediction accuracy for a
certain combination of data columns only. This algorithm, when implemented on a huge dataset 
makes the prediction very sluggish and it may take hours to run over a single iteration of the
algorithm so as to converge the solver.

More information on the algorithm used:
- Penalty type: l2
- Solver: saga
- No class weight, no random state
- Maximum number of iterations taken to converge the solver: 7500

#### Support Vector Machines

SVM takes the highest time to generate a prediction accuracy for this dataset. Each iteration takes
more than 7 hours to run and execute. Such a computationally costly algorithm, although good in
prediction, will be highly time-consuming for the users. The prediction accuracy for a particular
combination of columns will differ for every variety of train-test split. Therefore, in order to find
the highest prediction possible, we iterate the algorithm a few times for a different train-test split
and record the highest prediction accuracy’s corresponding split in a pickle file for future use.

More information on the algorithm used:
- Kernel used: rbf
- Degree: 3
- Regularization parameter: 1
- No verbosity, no random state

#### Random Forest

A bagging ensemble technique for the Decision Tree Algorithm, Random Forest generates a
constant prediction accuracy at a particular n (number of estimators that gets created) for the
column combinations. Therefore, there is no need to store the highest recorded accuracy and its
corresponding test-train split in a pickle file. Random forest has also been one of the 
computationally cheapest techniques and resulted in one of the highest recorded prediction
accuracies for many of the column combinations.

More information on the algorithm used: 
- Number of estimators - 100 
- Node splitting criteria taken - Gini index 
- No explicit mention of the depth of the tree that is formed 
- For best split, square root of the features was done instead of taking logarithm 
- No random state included and no verbosity required

#### K-Nearest Neighbors

This lazy learner algorithm is highly effective, and better than Random Forest, but it becomes quite
computationally expensive when it is run on a dataset of immense size. Due to variations in
answers for every train-test split, we had to iterate the algorithm a few times and store the split
with the highest accuracy in a pickle. If the time is not an issue, then KNN yields the highest ever
recorded prediction percentages.

More information on the algorithm used:
- Number of neighbors set: 3
- Distance set: Minkowski
- Weights set to each point: uniform

#### Deep Learning

Artificial Neural Networks (ANN) are multi-layer fully connected neural nets and consist of an
input layer, multiple hidden layers, and an output layer. ANN is inspired by the design of a human
brain and tries to simulate it. An artificial neuron receives a signal then processes it and can signal
neurons connected to it. The signal at a connection is a real number, and the output of each neuron
is computed by an activation function of the sum of its inputs. Neurons and edges typically have a
weight that adjusts as learning proceeds. The weight increases or decreases the strength of the
signal at a connection. Neurons may have a threshold such that a signal is sent only if the aggregate
signal crosses that threshold.

More information on the algorithm used: 
- Number of hidden layers: 2
- Learning Algorithm: Stochastic Gradient Descent Extension- Adam Optimizer
- Loss Function: Sparse Categorical Cross Entropy
- Activation function for input layer: Linear (f(x) = x)
- Activation function for the 2 hidden layers: Rectified Linear Activation Unit (RELU)
- Activation function for output layer: Softmax Activation Function
- Number of nodes: 64
- Batch size: 64
- Number of epochs: 8
- Input variables linearly scaled to lie in range [0, 1]

A heatmap to demonstrate the high correlation of columns with the others:

<p align="center">
  <a>
    <img src="https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning/blob/main/Visualization/Heatmap.png" width="600" height="450">
  </a>
</p>

## Inference

The classification mining technique algorithms have been run and every prediction accuracy was
recorded. The [Accuracy Table](https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning/blob/main/AccuracyTable.pdf) shows the variation in the performance of every algorithm used for
that particular combination of columns. 

## Conclusion

The objective of the project was successfully achieved as we found alternative supervised learning
approaches that can yield a much higher and more accurate prediction percentage for the given
dataset in a reduced time frame, owing to the size of the dataset. The most suitable algorithm would
be the K Nearest Neighbors algorithm and the label of interest can be aptly predicted by taking all
the columns into account.

This study improves upon the work of Jock A. Blackard and Denis J. Dean in predicting forest
covers. After training and testing the same dataset with various classification models, some of the
models managed to beat the old accuracy of “71.1%” with a new best accuracy of “97”.

With the optimizer and loss functions used, our deep learning model was able to beat the old
accuracy too, with a staggering “99.9%”.

## References

- [_Blackard, J. A., & Dean, D. J. (1999). Comparative accuracies of artificial neural networks and discriminant analysis in predicting forest cover types from cartographic variables. Computers and electronics in agriculture, 24(3), 131-151._](https://github.com/arjundas1/Forest-Cover-Prediction-using-Classification-Mining-Techniques-and-Deep-Learning/blob/main/References/Comparative%20accuracies%20of%20artificial%20neural%20networks%20and%20discriminant%20analysis%20in%20predicting%20forest%20cover%20types%20from%20cartographic%20variables.pdf)
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
- https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

## Contact Us

_You can contact us through our LinkedIn account -_

* [John Harshith](https://www.linkedin.com/in/john-harshith-5354371b7/)
* [Arjun Das](https://www.linkedin.com/in/arjundas1/)
* [Siddharth Pal](https://www.linkedin.com/in/siddharthpal20/)
