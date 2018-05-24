from keras import layers
from keras import models
from keras.models import load_model

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt 
import os 

"""
DESCRIPTION OF MODEL:
Basic model trained from scratch
No regularization or dropout of any sort, yet
Trainined on BIG images, no weighting of luminosity population
"""


# (A) Define parameters for model
input_shape = (128, 128)
source_data = "../../data_processing/lagos_p128_z1"
batch_size = 100
steps_per_epoch = 300
epochs = 30
validation_steps = 150
mod_name = "mod_1_BIG"
test_steps = 150


# (B) Build layers of model
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(input_shape[0], input_shape[1], 3)))
model.add(layers.MaxPooling2D( (2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D( (2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D( (2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D( (2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# (C) Optimization
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


# (D) Training/validation then save model to .h5 file
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = source_data + "/training"
validation_dir = source_data + "/validation"

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size= input_shape,
    batch_size = batch_size,
    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size= input_shape,
    batch_size = batch_size,
    class_mode = 'categorical')

history = model.fit_generator(
    train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, 
    validation_data=validation_generator, validation_steps=validation_steps)

save_to = "output/{}/".format(mod_name)

if not os.path.exists(save_to):
    os.makedirs(save_to)

model.save(save_to+'model.h5')

# (E) Save out graphs of training/validation loss & accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.savefig(save_to + 'performance.png')

# (F) Save a summary of the model structure along with performance
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_steps)

with open(save_to + 'structure.txt', 'w') as f:

    s = model.summary()
    f.write(s)
    f.write('\n\n')
    f.write("TEST ACCURACY: {}\n".format(test_acc))
    f.write("TEST LOSS: {}\n".format(test_loss))




# (F) Discard the last fully-connected layer
    # F.1 - calculate the feature vectors for each of the DHS observations
    # F.2 - run a regression of wealth index ~ feature vectors
