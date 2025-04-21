from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode="fine")

print("The training features' shape is", X_train.shape)
print("The training targets' shape is", y_train.shape)
print("The test features' shape is", X_test.shape)
print("The test targets' shape is", y_test.shape)

class_names = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle',
               'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar',
               'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile',
               'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
               'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
               'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter',
               'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
               'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
               'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
               'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
               'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

plt.figure(figsize=[10,10])
for i in range (30):
  plt.subplot(5, 6, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(X_train[i])
  plt.xlabel(class_names[y_train[i,0]])
plt.show()

X_train = X_train/255
X_test = X_test/255

n_classes = 100
print("Shape before one-hot encoding: ", y_train.shape)
Y_train = to_categorical(y_train, n_classes)
Y_test = to_categorical(y_test, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), strides=(1,1),
                 padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.35))

model.add(Conv2D(100, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(600, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(100, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X_train, Y_train, batch_size=64, epochs=10, validation_data=(X_test, Y_test))

plt.figure()
plt.plot(history.history['loss'], 'black', linewidth=2.0)
plt.plot(history.history['val_loss'], 'green', linewidth=2.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=12)
plt.show()

plt.figure()
plt.plot(history.history['accuracy'], 'black', linewidth=2.0)
plt.plot(history.history['val_accuracy'], 'blue', linewidth=2.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=14)
plt.xlabel('Epochs', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.title('Accuracy Curves', fontsize=12)
plt.show()

pred = model.predict(X_test)
pred_classes = np.argmax(pred, axis=1)

plt.figure(figsize=[10,10])
for i in range (30):
    plt.subplot(5, 6, i+1).imshow(X_test[i])
    plt.subplot(5, 6, i+1).set_title("True: %s \nPredict: %s" %
                      (class_names[y_test[i, 0]],
                       class_names[pred_classes[i]]))
    plt.subplot(5, 6, i+1).axis('off')

plt.subplots_adjust(hspace=1)
plt.show()

failed_indices = []
idx = 0
for i in y_test:
    if i != pred_classes[idx]:
        failed_indices.append(idx)
    idx = idx + 1

plt.figure(figsize=[10,10])
for i in range (30):
    random_select = np.random.randint(0, len(failed_indices))
    failed_index = failed_indices[random_select]
    plt.subplot(5, 6, i+1).imshow(X_test[failed_index])
    plt.subplot(5, 6, i+1).set_title("True: %s \nPredict: %s" %
                      (class_names[y_test[failed_index, 0]],
                       class_names[pred_classes[failed_index]]))
    plt.subplot(5, 6, i+1).axis('off')

plt.subplots_adjust(hspace=1)
plt.show()
