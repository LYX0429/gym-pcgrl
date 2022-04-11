import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import random

class ICM():
    
    def __init__(self, n, m):
        self.model = Sequential()
        self.model.add(Dense(units=256, input_shape=(n * m + 3, )))
        self.model.add(Dense(units=288,))
        self.model.add(Dense(units=n*m,))
        
        self.model.compile(optimizer='SGD',
                      loss='mse',
                      metrics=['accuracy'])
    
    def predict(self, state, action, next_state):
        x = state.flatten()
        x = np.append(x, action)
        x = np.array([x])
        y = np.array([next_state.flatten()])
        history = self.model.fit(x, y, epochs=10,verbose=0,)
        reward = history.history['loss'][0]
        return reward

if __name__ == '__main__':
    n = 5
    m = 5
    
    icm = ICM(n, m)
    
    dataSet = []
    for _ in range(100000):
        data = []
        for i in range(n):
            for j in range(m):
                data.append(random.random() / 2)
        dataSet.append(data)
    dataSet = np.array(dataSet)
    
    valSet = []
    for x in range(100000):
        val = dataSet[x] * 2
        valSet.append(val)
    valSet = np.array(valSet)
    
    icm.model.fit(dataSet, valSet, epochs=50, validation_split=0.3)
    icm.model.evaluate(np.array([dataSet[0]]),  np.array([valSet[0]]), verbose=2)
    pre = icm.model.predict(np.array([dataSet[0]]))
    print(pre)
    print(valSet[0])
    print(pre - valSet[0])


