import os
import numpy as np
import keras
import os
import tensorflow as tensorflow
import random as rn
from PIL import Image
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from keras import optimizers
from keras.models import Sequential, Model, load_model, model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing import image
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from matplotlib import pyplot as plt

train_files_num = 500
test_files_num = 500
img_width, img_height = 48, 48
train_path = '/Users/abhishek.venkatesh/Desktop/Capstone/Data/train_images'
test_path = '/Users/abhishek.venkatesh/Desktop/Capstone/Data/test_images'
batch_size = 20
epochs = 10
num_classes = 2


model = keras.applications.vgg16.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
for layer in model.layers:
    layer.trainable = False

x = model.output
x = Flatten()(x)
predictions = Dense(num_classes, activation="softmax")(x)

# creating the final model
model_final = Model(input = model.input, output = predictions)

model_final.summary()

# compile the model
model_final.compile(loss = "categorical_crossentropy", 
                    #optimizer=Adam(lr=.0001),
 			    	#optimizer='rmsprop',
 			    	optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), 
                    metrics=["accuracy"])

train_batches = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=5)

test_batches = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.1,
width_shift_range = 0.1,
height_shift_range=0.1,
rotation_range=5)

train_generator = train_batches.flow_from_directory(
train_path,
target_size = (img_height, img_width),
batch_size = batch_size,
class_mode = "categorical")

test_generator = test_batches.flow_from_directory(
test_path,
target_size = (img_height, img_width),
class_mode = "categorical")

checkpoint = ModelCheckpoint("car_model.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto')

fit_generator = model_final.fit_generator(
train_generator,
samples_per_epoch = train_files_num,
epochs = epochs,
validation_data = test_generator,
nb_val_samples = test_files_num,
callbacks = [checkpoint, early])

model_final.save('car_model.h5')
model_final.save_weights('car_model_weights.h5')






# vgg16_model = keras.applications.vgg16.VGG16()
# model = Sequential()
# for layer in vgg16_model.layers:
#  	model.add(layer)
# model.layers.pop()
# for layer in model.layers:
#  	layer.trainable = False
# model.add(Dense(2, activation='softmax'))
# # hidden = Dense(2, activation='softmax')(model.get_output_at(-2)).model.get_output_at(-1)
# model.summary()

# model.compile(#optimizer=Adam(lr=.0001),
# 			    #optimizer='rmsprop',
# 			    optimizer = optimizers.SGD(lr=0.0001, momentum=0.9),
# 				#optimizer=sgd,
# 				loss='categorical_crossentropy', 
# 				metrics=['accuracy']) 
# # steps_per_epoch = # in train_batches / batch_size 
# model.fit_generator(train_batches, steps_per_epoch=5, validation_data=valid_batches, validation_steps=5, epochs=5, verbose=2) 

# # imgs, labels = next(test_batches)
# # test_imgs, test_labels = next(test_batches)
# # test_labels = test_labels[:,0]
# # steps = # in test_batches / batch_size 
# predictions = model.predict_generator(test_batches, steps=1, verbose=0)
# # cm = confusion_matrix(test_labels, np.round(predictions[:,0]))
# # cm_plot_labels = ['cat', 'dog']
# # plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')








# cwd = os.getcwd()
# folder = '/Users/abhishek.venkatesh/Desktop/Capstone/Data/train_images'
# for sub_folder in os.listdir(folder):
#     path, dirs, files = next(os.walk(os.path.join(folder,sub_folder)))
#     train_files_num += len(files)


# folder = '/Users/abhishek.venkatesh/Desktop/Capstone/Data/test_images'
# for sub_folder in os.listdir(folder):
#     path, dirs, files = next(os.walk(os.path.join(folder,sub_folder)))
#     test_files_num += len(files)

# print(train_files_num,test_files_num)


# # CONFUSION MATRIX
# # %matplotlib inline
# from sklearn.metrics import confusion_matrix
# import itertools
# import matplotlib.pyplot as plt

# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
# 	# This function prints and plots the confusion matrix
# 	# Normalization can be applied by setting 'normalize=True'
# 	plt.imshow(cm, interpolation='nearest', cmap=cmap)
# 	plt.title(title)
# 	plt.colorbar()
# 	tick_marks = np.arange(len(classes))
# 	plt.xticks(tick_marks, classes, rotation=45)
# 	plt.yticks(tick_marks, classes)

# 	if normalize:
# 		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# 		print("Normalized confusion matrix")
# 	else:
# 		print("Confusion matrix, without normalization")

# 	print(cm)

# 	thresh = cm.max() / 2
# 	for i, h in itertools.product(range(cm.shahpe[0]), range(cm.shape[i])):
# 		plt.text(j, i, cm[i, j],
# 				 horizontalalignment="center",
# 				 color="white" if cm[i, j] > thresh else "black")

# 		plt.tight_layout()
# 		plt.ylabel('True label')
# 		plt.xlabel('Predicted label') 

# def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
# 	if type(ims[0]) is np.ndarray:
# 		ims = np.array(ims).astype(np.uint8)
# 		if (ims.shape[-1] != 3):
# 			ims = ims.transpose((0,2,3,1))
# 	f = plt.figure(figsize=figsize)
# 	cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
# 	for i in range(len(ims)):
# 		sp = f.add_subplot(rows, cols, i+1)
# 		sp.axis('Off')
# 		if titles is not None:
# 			sp.set_title(titles[i], fontsize=16)
# 		plt.imshow(ims[i], interpolation=None if interp else 'none')








# # print(fit_generator.history.keys())
# # plt.plot(fit_generator.history['acc'])
# # plt.plot(fit_generator.history['val_acc'])
# # plt.title('model accuracy')
# # plt.ylabel('acc')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'test'], loc='upper left')
# # plt.show()

# # plt.plot(fit_generator.history['loss'])
# # plt.plot(fit_generator.history['val_loss'])
# # plt.title('model loss')
# # plt.ylabel('loss')
# # plt.xlabel('epoch')
# # plt.legend(['train', 'test'], loc='upper left')
# # plt.show()












