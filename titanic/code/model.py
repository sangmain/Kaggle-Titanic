import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt

train = pd.read_csv("test1.csv")

data = train.copy()

end_prices = train['종가']

#normalize window
seq_len = 50
sequence_length = seq_len + 1

result = []
for index in range(len(end_prices) - sequence_length):
    idk = []
    idk[:] = end_prices[index: index + sequence_length]
    result.append(idk)


#normalize data
def normalize_windows(data):
    normalized_data = []
    for window in data:
        normalized_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalized_data.append(normalized_window)
    return np.array(normalized_data)

result = normalize_windows(result)

# split train and test data
row = int(round(result.shape[0] * 0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))

model.add(LSTM(64, return_sequences=False))

model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='rmsprop')

model.summary()

model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=10, epochs=20)

pred = model.predict(x_test)
print(pred)
print(x_test[54])
fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')

ax.legend()
plt.show()


model.save("model.h5")