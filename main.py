import os
import random
import numpy as np
from tensorflow.keras import utils
from tensorflow.keras import Sequential
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image


model = Sequential()
pretrained_resnet50 = ResNet50(include_top=False, weights='imagenet', pooling='max', input_shape=(224, 224, 3))

# Freeze the pretrained layer weights
for layer in pretrained_resnet50.layers:
    layer.trainable = False

model.add(pretrained_resnet50)
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))
model.summary()

adam = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

folders = os.listdir('Train')
image_data = []
labels = []
count = 0

for ix in folders:
    path = os.path.join("Train", ix)

    for im in os.listdir(path):
        try:
            img = image.load_img(os.path.join(path, im), target_size=(224, 224))
            img_array = image.img_to_array(img)
            image_data.append(img_array)
            labels.append(count)
        except Exception as e:
            print(e)
            pass
    count += 1

combined_dataset = list(zip(image_data, labels))
random.shuffle(combined_dataset)
image_data[:], labels[:] = zip(*combined_dataset)

X_train = np.array(image_data)
Y_train = np.array(labels)
Y_train = utils.to_categorical(Y_train)

model.fit(X_train, Y_train, batch_size=16, epochs=25, validation_split=0.20)
