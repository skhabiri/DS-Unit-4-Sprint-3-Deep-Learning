# CNN
### Image data
A monochrome image is of one color represented by 0 to 255. A gray scale is an example of monochrome with black 0 and white 255. scaling up or down the numbers does not change the information. For a colored image we do need to mix three colors of red, blue and green, which are primary colors of the most effective additive color system. To implement that we use three channels. Each channel is a monochrom of one of those colors.
### Convolutional Neural Networks
In ANN all the input samples are related to each other equally through a densely connected network. For images the nearby pixels are more related to each other than the pixels that are far apart. CNN instead of equally connected weights uses a kernel to process the neighboring pixels locally. Hence for image processing we prefer CNN rather than regular ANN. That would allow to extract features such as edge, face, sharpness from the image. different features would require different kernels or filters for extraction. Additionally the filters can be stack in parallel as well as in series to model more dimensions. The convolution operation starts by placing the weighted kernel at one corner of the image and translating the overlap window to a single number by the weighted sum of the numbers. By striding to the right and then down the entire image is going to be scanned. In convolution layer we usually stride with steps of 1 and in pooling we stride we bigger steps as we intend to reduce the size of the outcome. Similar to ANN in CNN we train the filter weights. One approach is to extract some features with the convolutional layer and then use pooling to zoom into certain features and apply convolutional layers again to extract detailed information on focused objects and repeat this process. After each convolutional layer we have an activation ReLu.
* Padding
Convolving an image with a kernel reduces the size of the output image and leads to loss of information. To formulize that, if the input image is of size 𝑛𝑥𝑛 and the kernel is 𝑓𝑥𝑓 and stride stepping is 𝑠, the output image is of size of one for the initial placement plus the remaining strides left to take. The outcome size is round down of: 
(1+(𝑛−𝑓)/𝑠) × (1+(𝑛−𝑓)/𝑠)
To avoid the loss of information, the surrounding of the image is padded with selected numbers. Most commonly with 0. Alternatively, it could be padded with the nearby values or average of them. To bring the size of the outcome to that of the original image, the total columns of padding, 𝑝, in one direction (ex/ both left and right padding columns) needs to be:
𝑝=𝑛(𝑠−1)+𝑓−𝑠
* Pooling
The initial input samples are typically large, let's say 254x254=65K. Adding parallel filters would dramatically increase the number of parameters. To handle this computing challenge pooling technique is applied which is basically extracting more important features in the first pass and zooming in on those with subsequent convolutional layers to get more relevant details instead of getting details from everything that might not be relevant to begin with. There are different pooling filters for different tasks. The most common one is max pooling which takes the maximum number in a pooling window as the most important feature. In pooling we take longer strides (typically non overlapping to help reduce the size. Pooling helps to speed up computation while it makes some of the features it detects a bit more robust.
* Output size for conv and max pooling layers:
Both conv and max pooling layers work with
(1+(𝑛+𝑝−𝑓)/𝑠) × (1+(𝑛+𝑝−𝑓)/𝑠)
Here p is the total number of padding columns/rows. f is the filter size, n is the input data size and s is number of strides. It seems keras ignores the last fractional column or rows where the filter does not fully overlap. Hence if the calculation is not a whole number you need to round it down.
* 1D, 2D, 3D Convolutional layer:
This simply refers to the dimension of convolutional operation. In other words the dimensions that the filter scans over. If the scan is along a line from left to right it's 1D layer. If the scan is additionally from top to bottom in a x-y coordinate, then it's a 2D Conv layer. In that sense a color image with three channels is still considered 2D, as there is no useful information to scan along the channels, or going from one channel to another. However for a video there are x-y image as well as the sequence of images in time which contains information about the content of the input. Hence scanning along the time adds a third dimension and needs a 3D convolutional layer.
* Convolution Operation on Volume
When the input has more than one channel (e.g. an RGB image), the filter should have a matching number of channels. To calculate one output cell, perform convolution on each matching channel, then add the result together.
* Convolution Operation with Multiple Filters
Multiple filters can be used in a convolution layer to detect multiple features. The output of the layer then will have the same number of channels as the number of filters in the layer.
* One Convolution Layer
Finally to make up a convolution layer, a bias (ϵ R) is added and an activation function such as ReLU or tanh is applied.
* Flattening
Once the data is processed with consecutive convolutional and pooling layers we flatten the data into a giant one dimensional array and connect it to a fully connected dense layer as it's more suitable for the final task of classification.
* Overall Architecture:¶
As the layers go deeper and deeper, the features that the model deals with become more complex. For example, at the early stage of ConvNet, it looks up for oriented line patterns and then finds some simple figures. At the deep stage, it can catch the specific forms of objects and finally be able to detect the object of an input image.
### Convolution with scipy
We use `scipy.ndimage.convolve()` to apply convolution on an image array manually. depending on the kernel we can detect the horizontal edge, vertical edge, increase sharpness and so on.
### Image classification:
Dataset: cifar10 with 50K training and 10K test. The image shape is 32x32x3 ndarray, with 10 classes.
Model: It’s a 8 layer sequential model implemented with keras API.
```
# Build Model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', strides=1,
                 input_shape=(32,32,3), data_format='channels_last', padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid', data_format='channels_last'))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Fit Model default batch_size=32
model.fit(train_images, train_labels, epochs=8, validation_data=(test_images, test_labels))
```
0.7 validation accuracy after 8 epochs with batch_size=32.
## Transfer Learning:
TensorFlow Hub: This lets you quickly take advantage of a model that was trained with thousands of GPU hours. It also enables transfer learning - reusing a part of a trained model (called a module) that includes weights and assets, but also training the overall model some yourself with your own data. The advantages are fairly clear - you can use less training data, have faster training, and have a model that generalizes better.
Keras API - Applications: Keras Applications are deep learning models that are made available alongside pre-trained weights. These models can be used for prediction, feature extraction, and fine-tuning. We'll focus on an image classifier: **ResNet50**.

### ResNet50: 
It’s a 50 layer deep residual learning network trained to recognize 1000 objects as an image classifier. It has a preprocess_input, decode_predictions module that interfaces with the model. The input image needs to be resized to 224x224 and converted to two dimensional np.ndarray before being fed into the preprocess_input module. The output of the model is also np.ndarray that after being decoded changes to a list of tuples with (internal_param, name_string, probability).
We use `requests` and `BeautifulSoup` to connect to a url and download a series of images. The purpose of the work is to determine if the image is a banana or not. We use the model out of the box without further training.
```
# resize the image for resNet50
x = preprocess_input(x)

# Instantiate the pre trained model
    model = ResNet50(weights='imagenet')

# Run prediction, type(features)=<class 'numpy.ndarray'>
features = model.predict(x)
    
# Post process resNet50 prediction results
results1 = decode_predictions(features, top=3)
```

## Notebook cnn-432a.ipynb: 
In this notebook we take a few hundred images of mountains and forests and do a binary classification. We use the image processing layers of ResNet50 and use Keras functional API to add the top layers and train the top layer parameters with our own data for classification. Using ImageDataGenerator allows us to label our classes with integer numbers. Later on when the model output is predicted for an image, it will be a continuous number in the range of classes that we can digitize it into discrete classes as the final step of our prediction.

### ImageDataGenerator: 
To input the data into the convolutional network we use ImageDataGenerator class. It scaled the data. It can read the data directly from directory and store them shuffled and in batches and at the same time resized to desired target size. It also creates an integer target label based on the number of classes.
```
train_image_generator = ImageDataGenerator(rescale=1./255)
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
```
### Instantiate Model: 
We take off the flatten and all the dense top layers from the model to customize it with our data. However, we also disable weight update for all the ResNet50 remaining layers to avoid over writing them while the update will take place on our customized top layers. GlobalAveragePooling2D is used instead of the `flatten` layer before connecting to the top layers. We use Keras functional API, Model, instead of the Sequential model to build the model.
```
resnet = ResNet50(weights='imagenet', include_top=False)
x = resnet.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(resnet.input, predictions)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
After training the model we get val_accuracy of 0.84.

### Custom CNN Model: 
Instead of using ResNet50 as the base we are building a custom CNN model from scratch. We would also use ModelCheckpoint callback to save the model with the best validation accuracy.

### How to interpret the output log metrics
1. *Since ImageDataGenerator has constructed the input data in batches, do not specify the steps or batch_size for training or validation data.*
2. *`accuracy` in the output log file of model.fit() is calculated once for each batch of epoch. The last one that is recorded in history dictionary and output log per epoch, is indeed the accuracy of the **last batch** of the training data for that epoch. Note that the batches are selected randomly. at each epoch*
3. *`val_accuracy` in the output log file or history dictionary of model.fit(), is only calculated once at the end of each epoch. After the last batch of an epoch, the entire validation data is run through the model to get the validation score in that epoch.
4. `model.evaluate(val_data)` should return the same metric as the validation metric of the last epoch.
4. `model.evaluate(train_data)` is different from the one in history log. This is for the entire train set not for a batch in an epoch.

### Build and compile
We build a 9 layer cnn model as follows.
```
model1 = Sequential()
model1.add(Conv2D(filters=32, kernel_size=(3,3), strides=1, padding='valid', activation='relu', 
input_shape=(224,224,3)))
model1.add(MaxPooling2D(pool_size=(2,2), strides=None, padding='valid'))
model1.add(Conv2D(32, (3,3), activation='relu'))
model1.add(MaxPooling2D((2,2)))
model1.add(Conv2D(64, (3,3), activation='relu'))
model1.add(Flatten())
model1.add(Dense(64, activation='relu', 
kernel_initializer=initializers.RandomNormal(stddev=0.01), 
bias_initializer=initializers.Zeros()))
model1.add(Dropout(0.2))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(optimizer='nadam', loss='binary_crossentropy', metrics=['accuracy'])
```

### Callback to save the best epoch model instead of the last epoch
We use ModelCheckpoint callback to save the best model and load it later, and also use a custom callback to save the training accuracy after each batch.
```
checkpoint_cb = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=2, 
save_best_only=True, mode='auto')

class BatchHistory(Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = [[] for _ in range(34)]

    def on_batch_end(self, batch, logs={}):
        self.accuracy[batch].append(logs.get('accuracy'))
batchhistory_cb = BatchHistory()
```
### Fit and evaluate
evaluate() returns loss and accuracy metrics while predict returns each classes' continuous numeric representation.

```
history = model1.fit(train_data_gen, epochs=10, validation_data=val_data_gen, 
callbacks=callbacks_list)

best_model = tf.keras.models.load_model("./best_model/best-02-0.9282.hf5")
best_model.evaluate(val_data_gen)
pred = model1.predict(val_data_gen)
```

## Image Data Augmentation
To simulate an increase in a sample of image, you can apply image manipulation techniques: cropping, rotation, stretching, etc. Luckily Keras has some handy functions for us to apply these techniques to our mountain and forest example. In each epoch, the ImageDataGenerator applies a transformation on the images you have and uses the transformed images for training. The set of transformations includes rotation, zooming, etc. By doing this you're somehow creating new data (i.e. also called data augmentation), but obviously the generated images are not totally different from the original ones. This way the learned model may be more robust and accurate as it is trained on different variations of the same image. We need to emphasize that the model is trained on a random augmented version of the same training images which is going to vary from one epoch to another. We use ImageDataGenerator for the problem.

Instantiate the ImageDataGenerator class
```
train_image_generator1 = ImageDataGenerator(
        rotation_range=20,  # Degree range for random rotations.
        rescale=1./255,
        shear_range=0.1,  # dragging the image
        brightness_range=[0.5,1.0],
        zoom_range=(0.8, 1.2),
        horizontal_flip=True,
        vertical_flip = False,
        fill_mode='nearest')
```

Generate augmented image data
```
train_data_gen1 = train_image_generator1.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           seed=110,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')
```
**Image data generator dimensions:**
- batch number
    - X, y
- X (batch_size, row, column, channel), y (batch_size,)

Using the augmented epochs we get val_accuracy of 0.87

### Libraries:
```
from IPython.display import YouTubeVideo
from IPython.display import Image as ipyImage
import imageio
from skimage import color, io
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from skimage.measure import block_reduce
import PIL
from PIL import Image as pilImage
from matplotlib import image as matimage
from matplotlib import pyplot
from tensorflow.keras.preprocessing import image as kimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import numpy as np
import scipy.ndimage as nd
from tensorflow.keras import datasets
from tensorflow.keras.models import Sequential, Model  # This is the functional API
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


from bs4 import BeautifulSoup
import requests
import os, random
```
