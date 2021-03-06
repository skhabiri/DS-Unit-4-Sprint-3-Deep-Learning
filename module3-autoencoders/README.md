

# Autoencoder
An autoencoder is a type of artificial neural network used to learn efficient data codings in an unsupervised manner. The aim of an autoencoder is to learn a representation (encoding) for a set of data, typically for dimensionality reduction, by training the network to ignore “noise”. Along with the reduction side, a reconstructing side is learnt, where the autoencoder tries to generate from the reduced encoding a representation as close as possible to its original input. The encoder compresses the input data and the decoder does the reverse to produce the uncompressed version of the data to create a reconstruction of the input as accurately as possible. The learning process is described simply as minimizing a loss function:  𝐿(𝑥,𝑔(𝑓(𝑥))).
dataset:
First we use quickdraw10 to create a simple representation of the input images. We have 100K of images in 10 category.

### Simple Autoencoder
For the simple autoencoder we use fully connected dense layer and sigmoid activation layer. Essentially we take an input of 784 pixels normalized between 0 and 1 and output 784 neurons with continuous numerical output between 0 and 1. hence we use sigmoid activation function to have the output limited to 0 and 1.
```
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

autoencoder.compile(optimizer='nadam', loss='binary_crossentropy')
```
The reconstructed output is of a lower quality compared to the input as it's been down sampled.
Half the autoencoder is the encoder part that will later be utilized to lower the dimension of the input.
```
autoencoder.fit(x_train, x_train,
                epochs=10000,
                batch_size=64,
                shuffle=True,
                validation_split=.2,
                verbose = True,
                callbacks=[stop, tensorboard])
```
After fitting the model, we can call the predict method or the model directly to plot the reconstructed images. the output of the autoencoder() model is of type tensor and requires .numpy() method to be able to apply the reshape method for plotting. However, autoencoder.predict() outputs numpy ndarray.
```
decoded_imgs = autoencoder(x_train)
print(decoded_imgs.shape, type(decoded_imgs))

decoded_imgs_np = autoencoder.predict(x_train)
print(decoded_imgs_np.shape, type(decoded_imgs_np))

(100000, 784) <class 'tensorflow.python.framework.ops.EagerTensor'>
(100000, 784) <class 'numpy.ndarray'>
```
### Deep Autoencoder
By increasing the number of the dense layer we can reconstruct the image with smaller amount of loss.
```
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded) # this is the final encoded layer
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='nadam', loss='binary_crossentropy')
```
Here we are using all the dataset as training set as there is no test data required. However in the process of training we split the train to train and validation data to avoid overfitting.
```
autoencoder.fit(x_train, x_train,
                epochs=10000,
                batch_size=64,
                shuffle=True,
                validation_split=.2,
                verbose = True,
                callbacks=[stop, tensorboard])
```
Decoded images are available by `decoded_imgs = autoencoder.predict(x_train)`.

### Convolutional Autoencoder
Since our inputs are images, it makes sense to use convolutional neural networks (convnets) as encoders and decoders. In practical settings, autoencoders applied to images are always convolutional autoencoders --they simply perform much better. Let's implement one. The encoder will consist of a stack of Conv2D and MaxPooling2D layers (max pooling being used for spatial down-sampling), while the decoder will consist in a stack of Conv2D and UpSampling2D layers. In CNN since the padding parameter changes the tensor shape, depending on how we set the parameters the output tensor shape would differ. In general the autoencoder require that the input and output size to match and does not necessarily need a full symmetry from the center layer.
for conv2D we need 2D images not flatten `x_train = x_train.reshape((x_train.shape[0], 28, 28))`.
```
input_img = Input(shape=(28, 28, 1))
encoded = Conv2D(16, (3,3), activation='relu', padding='same')(input_img)
encoded = MaxPooling2D((2,2), padding='same')(encoded)
encoded = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
encoded = MaxPooling2D((2,2), padding='same')(encoded)
encoded = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
encoded =  MaxPooling2D((2,2), padding='same')(encoded)

decoded = Conv2D(8, (3,3), activation='relu', padding='same')(encoded)

decoded = UpSampling2D((2,2))(decoded)
decoded = Conv2D(8, (3, 3), activation='relu', padding='same')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(16, (3, 3), activation='relu')(decoded)
decoded = UpSampling2D((2, 2))(decoded)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)

autoencoder = Model(input_img, decoded)

autoencoder.fit(x_train, x_train,
                epochs=10000,
                batch_size=32,
                shuffle=True,
                validation_split=.2,
                verbose = True,
                callbacks=[stop, tensorboard])
```
We get better constructed images with CNN. We can plot the reduced dimension of the image to visualize the image representation in reduced dimension.
```
encoder = Model(input_img, encoded)
encoded_imgs = encoder.predict(x_train)

plt.imshow(encoded_imgs[10000].reshape(4*2, 4 * 4).T)
```

### Information Retrieval with Autoencoders
We train a knn model based on the flatten output of the encoder which is a reduced dimension version of the image and use that to find the label for any image based on the nearest neighbor (unsupervised learning). This is also called reverse image search.
```
encoded = tf.keras.layers.Flatten()(encoded)
encoder = Model(input_img, encoded)

encoded_imgs = encoder.predict(x_train)
encoded_imgs.shape
(100000, 128)
```
To fit the KNN:
```
from sklearn.neighbors import NearestNeighbors

nn = NearestNeighbors(n_neighbors=10, algorithm='ball_tree')
nn.fit(encoded_imgs)
```
and make a query on any image: 
```
query_encoding = encoder.predict(np.expand_dims(x_train[query],0))
results = nn.kneighbors(query_encoding)
# results[0] is the distance of neighbors from the query
# results[1] is the index of the neighbors t0 the query
```
If the query is a clock we can see that the predicted neighbors are clocks as well.
