
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf 
from keras.models import Sequential  #ANN
from keras.layers import Dense       #the layers in the ANN

#letters
letters = pd.read_csv('C:/Users/FM/.spyder-py3/data-set/emnist-letters-train.csv') 

orginal_letters = letters
letters=letters.drop('23',axis=1)

plt.imshow(np.reshape(letters.values[5],(28,28)),cmap="gray")
plt.show()

letters_test = pd.read_csv('C:/Users/FM/.spyder-py3/data-set/emnist-letters-test.csv')
orginal_letterstest = letters_test
letters_test = letters_test.drop('8',axis=1)


#ANN 
#build the model
model = Sequential()
model.add(Dense(64 , activation='relu', input_dim=784))     
model.add(Dense(64 , activation='relu'))
model.add(Dense(10 , activation='softmax'))



#compile the model #better accuracy
model.compile(
    optimizer = 'adam',
     loss='categorical_crossentropy', metrics=['accuracy'])

model.save_weights('letters.model')


plt.imshow(np.reshape(letters.values[1],(28,28)),cmap="gray")
plt.show()
plt.imshow(np.reshape(letters.values[2],(28,28)),cmap="gray")
plt.show()
plt.imshow(np.reshape(letters.values[3],(28,28)),cmap="gray")
plt.show()
plt.imshow(np.reshape(letters.values[4],(28,28)),cmap="gray")
plt.show()
#predict on the first 4 test images
predictions = model.predict(letters_test[:4])
print(predictions)
print(np.argmax(predictions, axis=1))
