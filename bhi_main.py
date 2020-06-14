from glob import glob
import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import imageio
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from PIL import Image
from collections import namedtuple
import random
# base parameters
numbers_of_file_to_use = 5000
test_size = 0.2  # part of whole dataset
epochs = 50
IMG_HEIGHT = 50
IMG_WIDTH = 50

# ---train or load----
build_and_fit_model = 1
save_model = 0
dir_to_save_model = r'saved_model\3_model_regul_drop'
dir_to_load_model = r'saved_model\first_model'

#  ----collect files paths----
files_paths_0 = glob(r'data\8863\0\*.png', recursive=True)
files_paths_1 = glob(r'data\8863\1\*.png', recursive=True)

# ----show example image----
# image = mpimg.imread(files_paths_0[1])
# plt.imshow(image, shape=(50, 50))
# plt.figure(figsize=(2, 2))
# image2 = imageio.imread(files_paths_1[1])
# plt.imshow(image2, shape=(50, 50))
# plt.xlabel('IDC (+)')
# plt.colorbar()
# image3 = np.array(Image.open(files_paths_0[1]).convert('L')) / 255
# Image.open(files_paths_0[1]).size
# plt.imshow(image3, shape=(50, 50), cmap='gray')
# plt.colorbar()
#
# image = imageio.imread(files_paths_0[1])
# plt.figure(figsize=(4, 4))
# plt.hist(image[:, :, 0].reshape(-1), bins=40, alpha=0.4, color='r', lw=0)
# plt.hist(image[:, :, 1].reshape(-1), bins=40, alpha=0.4, color='g', lw=0)
# plt.hist(image[:, :, 2].reshape(-1), bins=40, alpha=0.4, color='b', lw=0)
# plt.xlabel('IDC (-)')


# ----explore image dataset----
print('Load file list')
files = np.array(glob(r'data\8*\**\*.png', recursive=True))
files_0 = list(filter(lambda x: 'class0' in x, files))
files_1 = list(filter(lambda x: 'class1' in x, files))
all_samples_amount = len(files_0) + len(files_1)
negative_samples_amount = len(files_0)
positive_samples_amount = len(files_1)
print(f'Number of all samples: {len(files_0) + len(files_1)}')
print(f'Number of negative sample {len(files_0)}')
print(f'Number of positive sample {len(files_1)}')
print(f'Percent of negative samples: {round((negative_samples_amount / all_samples_amount) * 100, 2)}')
print(f'Percent of positive samples: {round((positive_samples_amount / all_samples_amount) * 100, 2)}')


# ----split dataset on test and training----
def split_and_randomize_data(file_list, files_amount, test_size):
    spliter = ShuffleSplit(n_splits=1, test_size=test_size)
    train_files_idx, test_files_idx = list(*spliter.split(file_list))
    shorter_train_files_idx = train_files_idx[:int(files_amount * (1 - test_size))]
    shorter_test_files_idx = test_files_idx[:int(files_amount * test_size)]
    return shorter_train_files_idx, shorter_test_files_idx


print('Split data on train and test datasets')
train_files_list_idx, test_files_list_idx = split_and_randomize_data(files, numbers_of_file_to_use, test_size=test_size)


processed_dataset = namedtuple('processed_dataset', ('labels', 'images'))
def get_images_dataset_as_dataframe(files_list: np.array, files_idx: np.array) -> processed_dataset():
    processed_dataset.labels = np.empty(0)
    processed_dataset.images = np.empty((0, IMG_WIDTH, IMG_HEIGHT, 3))
    for nr, path in enumerate(files_list[files_idx]):
        im = Image.open(path)# .convert('L') read in grayscale
        if im.size != (IMG_WIDTH, IMG_HEIGHT):
            continue
        label = 0 if 'class0' in path else 1
        processed_dataset.labels = np.append(processed_dataset.labels, [label], axis=0)
        image = np.array(im)
        processed_dataset.images = np.append(processed_dataset.images, [image], axis=0)
        if nr % 100 == 0 or nr == 0:
            print(f'Loaded files {nr} / {len(files_idx)}')
    print(f'Loaded {len(files_idx)}/{len(files_idx)}')
    return processed_dataset


print('Load images')
test_dataset = get_images_dataset_as_dataframe(files, test_files_list_idx)
train_dataset = get_images_dataset_as_dataframe(files, train_files_list_idx)
print(test_dataset.__name__, test_dataset._fields, 'with shape',  test_dataset.labels.shape, test_dataset.images.shape)
print(train_dataset.__name__, train_dataset._fields, 'with shape', test_dataset.labels.shape, train_dataset.images.shape)

# ----normalize data----
print('Normalize data')
train_dataset.images = train_dataset.images / 255
test_dataset.images = test_dataset.images / 255
# plt.figure()
# plt.imshow(train_dataset['images'][0])
# plt.colorbar()
# plt.grid(False)
# plt.show()


# plt.figure(figsize=(10, 10))
# for nr in range(9):
#     plt.subplot(3, 3, nr+1)
#     plt.grid(False)
#     plt.imshow(train_dataset['images'][nr])
#     plt.colorbar()
#     plt.xlabel(f'--IDC: {train_dataset["IDC"][nr]}--')


# ---- build model ----
if build_and_fit_model:
    print('Build model')

    # # simple dense
    model2 = keras.Sequential([
        keras.layers.Flatten(input_shape=(50, 50, 3)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2)
    ])
    #
    # # dense
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(50, 50, 3)),
    #     keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Dense(2)
    # ])

    model2.compile(optimizer='Adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # convolutional

    # model = keras.Sequential([
    #     keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    #     keras.layers.MaxPool2D(),
    #     keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    #     keras.layers.MaxPool2D(),
    #     keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     keras.layers.MaxPool2D(),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(512, activation='relu'),
    #     keras.layers.Dense(1)
    # ])

    # convolutional with dropout

    model = keras.Sequential([
        keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(0.2),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])


    model2.summary()
    print('Fit model')
    model.fit(train_dataset.images, train_dataset.labels, epochs=epochs, validation_data=(test_dataset.images, test_dataset.labels))

    if save_model:
        print('Save model')
        model.save(dir_to_save_model)


if not build_and_fit_model:
    print('Load model')
    model = tf.keras.models.load_model(dir_to_load_model)

print('Test model on test_dataset')
test_loss, test_acc = model.evaluate(test_dataset.images, test_dataset.labels, verbose=2)
print('\nTest accuracy:', test_acc)

# ----plot loss and accuracy----
loss_train_history = model.history.history['loss']
acc_train_history = model.history.history['accuracy']
loss_val_history = model.history.history['val_loss']
acc_val_history = model.history.history['val_accuracy']
epochs_list = model.history.epoch

plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_list, loss_train_history, label='loss_train')
plt.plot(epochs_list, loss_val_history, label='loss_val')
plt.legend()
#plt.xticks(epochs_list)
plt.subplot(1, 2, 2)
plt.plot(epochs_list, acc_train_history, label='acc_train')
plt.plot(epochs_list, acc_val_history, label='acc_val')
plt.legend()
#plt.xticks(epochs_list)

# ----explore/test model ---

# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
#
# predictions = probability_model.predict(test_dataset.images)
#
# font = {'family': 'serif',
#         'color': 'red',
#         'weight': 'normal',
#         'size': 12}
#
#
# def plot_image(image, label, prediction, font):
#     predicted_label = np.argmax(prediction)
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(image, cmap='gray')
#
#     if label == predicted_label:
#         font['color'] = 'green'
#     else:
#         font['color'] = 'darkred'
#
#     plt.xlabel(f'-IDC: {predicted_label}- with {100 * prediction[0]: .2f}%', fontdict=font)
#
#
# # draw example image with predicted and real data information
# # plot_image(test_dataset.images[0], test_dataset.labels[0], predictions[0], font)
#
# def plot_prediction_chart(label, prediction):
#     predicted_label = np.argmax(prediction)
#
#     plt.grid(False)
#     plt.xticks([0, 1])
#     current_plot = plt.bar((0, 1), height=predictions[0], width=0.4)
#     if label == predicted_label:
#         current_plot[predicted_label].set_color('green')
#     else:
#         current_plot[predicted_label].set_color('red')
#
#
# # draw example bar chart with predicted and real data information
# # plot_prediction_chart(0, predictions[0])
#
# print('Print example test images')
# amount_of_images_in_row = 3
# amount_of_images_in_column = 3
#
# random_images_idx = random.sample(range(len(test_dataset.labels)), amount_of_images_in_row*amount_of_images_in_column)
# plt.figure(figsize=(2*amount_of_images_in_column*2.5, 2*amount_of_images_in_row))
# for i, image_idx in enumerate(random_images_idx):
#     plt.subplot(amount_of_images_in_row, 2*amount_of_images_in_column, 2*i+1)
#     plot_image(test_dataset.images[image_idx], test_dataset.labels[image_idx], predictions[image_idx], font)
#     plt.subplot(amount_of_images_in_row, 2*amount_of_images_in_column, 2*i+2)
#     plot_prediction_chart(test_dataset.labels[image_idx], predictions[image_idx])

