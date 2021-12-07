# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

# importing dataset
dataset = pd.read_csv('../covtype.csv')

# Encoding dependent variable
X = dataset[dataset.columns[:55]]
scale = StandardScaler()
le = LabelEncoder()
y = le.fit_transform(list(dataset["Cover_Type"]))

# Train-Test splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalization
train_norm = X_train[X_train.columns[0:10]]
test_norm = X_test[X_test.columns[0:10]]

std_scale = StandardScaler().fit(train_norm)

X_train_norm = std_scale.transform(train_norm)
training_norm_col = pd.DataFrame(
    X_train_norm, index=train_norm.index, columns=train_norm.columns)
X_train.update(training_norm_col)

X_test_norm = std_scale.transform(test_norm)
testing_norm_col = pd.DataFrame(
    X_test_norm, index=test_norm.index, columns=test_norm.columns)
X_test.update(testing_norm_col)

# Creating ANN
cover_model = tf.keras.models.Sequential()

# Adding 1st layer
cover_model.add(tf.keras.layers.Dense(
    units=64, activation='relu', input_shape=(X_train.shape[1],)))
cover_model.add(tf.keras.layers.Dense(units=64, activation='relu'))
cover_model.add(tf.keras.layers.Dense(units=8, activation='softmax'))

# Adding output layer
cover_model.compile(optimizer=tf.optimizers.Adam(
), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_cover = cover_model.fit(
    X_train, y_train, epochs=8, batch_size=64, validation_data=(X_test, y_test))

# plotting accuracy graph
plt.plot(history_cover.history['accuracy'])
plt.plot(history_cover.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
