from keras import layers,models
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

#print("Train Image Shape: ",train_images.shape)
#print(len(train_images))
#print("Total Train Labels: ",len(train_labels))
#print(train_labels)

network=models.Sequential(name='New MnistModel')
#print(network.name)
network.add(layers.Dense(512,activation="relu",input_shape=(28*28,)))
network.add(layers.Dense(10,activation="softmax"))

network.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=['accuracy'])

# reshaping is necessary bcoz input image dimension shoulf be 2 instead of 3

train_images=train_images.reshape((60000,28*28))
train_images=train_images.astype('float32')/255
test_images=test_images.reshape((10000,28*28))
test_images=test_images.astype('float32')/255

# after reshaping it is necessart to cast labels to to_categorical for 1hotencoding

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

network.fit(train_images, train_labels,epochs=5,batch_size=128)

test_loss,test_acc=network.evaluate(test_images,test_labels)
print(test_loss)
print(test_acc)

digit = train_images[0]
print(digit.shape)
p=network.predict(test_images[0:1], verbose=1)
print(p)

index=np.argmax(p)
print(index)