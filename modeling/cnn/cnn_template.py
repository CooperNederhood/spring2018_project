from keras import layers
from keras import models
from keras.models import load_model

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import keras.applications as applications

import matplotlib.pyplot as plt 
import os 

"""
DESCRIPTION OF MODEL:

"""


# (A) Define parameters for model
input_shape = (34, 34)
source_data = "../../data_processing/[[SPECIFIC_FOLDER]]"
batch_size = 20
steps_per_epoch = 100
epochs = 30
validation_steps = 50
mod_name = "CNN NAME"
test_steps = 100

conv_base = applications.VGG16(weights='imagenet', include_top=False)
tune_layer_treshold = "block5_conv1"


# (B) Build layers of model on top of the convolutional base
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# (C) Freeze conv base, compile, and Optimization
conv_base.trainable = False
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])


# (D.a) Training/validation with the newly added dense layer then save model to .h5 file
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

model.save(save_to+'cnn_base.h5')

# (E) Save out graphs of training/validation loss & accuracy
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy - new FC layer')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss - new FC layer')
plt.legend()

plt.savefig(save_to + 'performance_base.png')

# (D.b) Unfreeze some of the conv base
conv_base.trainable = True 

set_trainable = False 
for layer in conv_base.layers:
    if layer.name == tune_layer_treshold:
        set_trainable = True 
    if set_trainable:
        layer.trainable = True 
    else:
        layer.trainable = False

# (D.c) Retrain the unfrozen base and new layer, with a slow learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])

history_final = model.fit_generator(
    train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, 
    validation_data=validation_generator, validation_steps=validation_steps)

model.save(save_to+'cnn.h5')

acc_final = history_final.history['acc']
val_acc_final = history_final.history['val_acc']

loss_final = history_final.history['loss']
val_loss_final = history_final.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc_final, 'bo', label='Training acc')
plt.plot(epochs, val_acc_final, 'b', label='Validation acc')
plt.title('Training and validation accuracy - finetune')
plt.legend()

plt.figure()

plt.plot(epochs, loss_final, 'bo', label='Training loss')
plt.plot(epochs, val_loss_final, 'b', label='Validation loss')
plt.title('Training and validation loss - finetune')
plt.legend()

plt.savefig(save_to + 'performance_full.png')


# (F) Save a summary of the model structure along with performance
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode='categorical')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=test_steps)

with open(save_to + 'structure.txt', 'w') as f:


    b = conv_base.summary()
    f.write(b)
    f.write('\n\n')

    s = model.summary()
    f.write(s)
    f.write('\n\n')
    f.write("TEST ACCURACY: {}\n".format(test_acc))
    f.write("TEST LOSS: {}\n".format(test_loss))




# (F) Discard the last fully-connected layer
    # F.1 - calculate the feature vectors for each of the DHS observations
    # F.2 - run a regression of wealth index ~ feature vectors
