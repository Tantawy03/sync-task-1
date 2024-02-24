import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras

path = "D:/internship/sync intern/archive (1)/plantvillage dataset/color"

train, test = keras.utils.image_dataset_from_directory(
    path,
    image_size=(224, 224),
    batch_size=32,
    seed=123,
    validation_split=.2,
    subset='both'
)

classes = train.class_names
classes

# Samples reveale

image1 = cv2.imread(
    "D:/internship/sync intern/archive (1)/plantvillage dataset/color/Cherry_(including_sour)___Powdery_mildew/1ca9be51-dea4-4075-8907-e583f85254b2___FREC_Pwd.M 4850.JPG")
plt.figure(figsize=(6, 6))
plt.imshow(image1)
plt.title('Cherry powdery mildew', size=18)
plt.axis('off')
plt.show()

image2 = cv2.imread(
    "D:/internship/sync intern/archive (1)/plantvillage dataset/color/Corn_(maize)___Northern_Leaf_Blight/0d0f6d14-be5c-4cb8-adb4-2cfd4d5f8540___RS_NLB 3642.JPG")
plt.figure(figsize=(6, 6))
plt.imshow(image2)
plt.title('Corn northern leaf blight', size=18)
plt.axis('off')
plt.show()

image3 = cv2.imread(
    "D:/internship/sync intern/archive (1)/plantvillage dataset/color/Pepper,_bell___Bacterial_spot/1b0cfb07-f452-49e0-85ad-45f3f519ca7a___JR_B.Spot 9094.JPG")
plt.figure(figsize=(6, 6))
plt.imshow(image3)
plt.title('pepper bell bacterial spot', size=18)
plt.axis('off')
plt.show()

# CNN
model = keras.Sequential([
    keras.layers.Rescaling(scale=1 / 255, input_shape=(224, 224, 3)),

    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Dropout(0.2),

    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPool2D((2, 2)),

    # fully connected layers

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(38, activation='sigmoid')

])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics='accuracy'
)

model.summary()

# Model training

history = model.fit(train, epochs=5)

accuracy = history.history['accuracy']
loss = history.history['loss']
epochs = range(1, 21)

plt.plot(epochs, accuracy, label='Acuuracy')
plt.plot(epochs, loss, label='loss')
plt.legend()
plt.show()

model.evaluate(test)


def img_to_pred(image):
    image = image.numpy()
    image = tf.expand_dims(image, 0)
    return image


plt.figure(figsize=(18, 18))
for images, labels in test.take(1):  # take the first patch
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        plt.imshow(images[i].numpy().astype('uint32'))
        plt.axis('off')
        actual = classes[labels[i]]
        predict = classes[np.argmax(model.predict(img_to_pred(images[i])))]
        plt.title(f"actual : {actual}  \n predicted : {predict} ")
