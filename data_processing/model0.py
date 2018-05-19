from keras import layers
from keras import models

'''
From Chapter 5, create a simple CNN model from scratch
'''

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)))
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

from keras import optimizers
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = "lagos/training"
validation_dir = "lagos/validation"

train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(128, 128),
	batch_size=20,
	class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
	validation_dir,
	target_size=(128, 128),
	batch_size=20,
	class_mode='categorical')

history = model.fit_generator(
	train_generator, steps_per_epoch=100, epochs=30, 
	validation_data=validation_generator, validation_steps=50)

model.save('first_lum_model.h5')
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
