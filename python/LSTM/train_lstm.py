import numpy as np
import pandas as pd

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split

noaction_dataFrame = pd.read_csv("./files/noaction.csv")
picking_dataFrame = pd.read_csv("./files/picking.csv")
stealing_dataFrame = pd.read_csv("./files/stealing.csv")

window_size = 5

data_frames = [noaction_dataFrame,picking_dataFrame, stealing_dataFrame]
number_of_classes = len(data_frames)

X = []
y = []
for i, df in enumerate(data_frames):
    dataset = df.iloc[:, 1:].values
    n_sample = len(dataset)
    for j in range(window_size, n_sample):
        X.append(dataset[j - window_size : j, :])
        y.append(i)
    

X, y = np.array(X), np.array(y)
print("X.shape, y.shape:", X.shape, y.shape)

# One-hot encoding
y = to_categorical(y, num_classes=number_of_classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(window_size, X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(50, activation="relu"))
# model.add(Dense(units=1, activation="sigmoid"))
model.add(Dense(units=number_of_classes, activation="softmax"))


# model.compile(optimizer="adam", metrics=["accuracy"], loss="binary_crossentropy")
model.compile(
    optimizer=Adam(learning_rate=0.001),
    metrics=["categorical_accuracy"],
    loss="categorical_crossentropy",
)  # for multiple

model.fit(X_train, y_train, epochs=24, batch_size=32, validation_data=(X_test, y_test))
model.save("./results/lstm01.keras")


# DOCUMENTS BY ME

# loss="categorical_crossentropy" >> we need y = to_categorical(y, num_classes=number_of_classes) 
# if loss="sparse_categorical_crossentropy", y.append(i) is enough

# metric = "categorical_accuracy" (compare with its class)"
# metric = "accuracy" (compare with all classes)"