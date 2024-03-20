import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

drinking_dataFrame = pd.read_csv("./files/drinking.txt")
standing_dataFrame = pd.read_csv("./files/standing.txt")

X = []
y = []
time_steps = 10

dataset = drinking_dataFrame.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(time_steps, n_sample):
    X.append(dataset[i - time_steps : i, :])
    y.append(0)
    
dataset = drinking_dataFrame.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(time_steps, n_sample):
    X.append(dataset[i - time_steps : i, :])
    y.append(1)

dataset = standing_dataFrame.iloc[:, 1:].values
n_sample = len(dataset)
for i in range(time_steps, n_sample):
    X.append(dataset[i - time_steps : i, :])
    y.append(2)

X, y = np.array(X), np.array(y)
print("X.shape, y.shape:", X.shape, y.shape)


# One-hot encoding for multi-class classification
y = to_categorical(y, num_classes=3)  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(time_steps, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

# model.add(Dense(units=1, activation="sigmoid")) 
model.add(Dense(units=3, activation="softmax")) # for multiple


# model.compile(optimizer="adam", metrics=["accuracy"], loss="binary_crossentropy") 
model.compile(optimizer="adam", metrics=["accuracy"], loss="categorical_crossentropy") # for multiple

model.fit(X_train, y_train, epochs=24, batch_size=32, validation_data=(X_test, y_test))
model.save("./results/lstm01.keras")
