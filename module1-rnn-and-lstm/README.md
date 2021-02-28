# RNN and LSTM
A time series is data where you have not just the order but some actual continuous marker for where they lie "in time" - this could be a date, a timestamp, Unix time, or something else. All time series are also sequences, and for some techniques you may just consider their order and not "how far apart" the entries are (if you have particularly consistent data collected at regular intervals it may not matter). The purpose of this module is to understand how data varies over time (or any sequential order), and use the order/time dimension predictively.

### RNN; A Neural Network for Sequences
RNN stands for Recurrent Neural Network. An LSTM is a Long Short-Term Memory layer, which is a type of RNN. These layers, built into a network, function as some sort of memory that allows the network to infer from not only the present, but also past events. The original RNN layer would tend to overwrite too much of its “internal memory” at each step, losing the ability to infer from events further in the past. The LSTM layer architecture is instead built in such a way that the network “decides” whether to modify its “internal memory” at each step. Doing so, and if properly trained, the layer can keep track of important events from further in the past, allowing for much richer inference. advantage (ane namesake) of LSTM is that it can generally put more weight on recent (short-term) events while not completely losing older (long-term) information. LSTM is widely used in NLP as  language is inherently ordered data (letters/words go one after another, and the order matters).

### LSTM Layer Architecture
The input sequences have S time steps, and each time step has C features. For each time step we have a LSTM cell that includes D parallel hidden units. The input tensor to the LSTM layer is [batch_size, step_times, features]. The output tensor of the LSTM has [batch_size, units]. Time steps are in series and do not show in the output interface, or generate any parameter to train.

### RNN/LSTM Sentiment Classification with Keras
dataset: It’s 25K imbd review from keras. Each review is tokenized words from a dictionary of 20K words. We have a 25K labeled train set and 25K test set. The purpose is to find if the sentiment of the review is positive (1) or negative (0).
steps to follow:
Read 25K imbd reviews, with a bag of 20K words as our dictionary of words >> x_train
each word is represented by an ordinal integer index
use sequence.pad_sequences to fix the size of each review to 80 words
In building the model use tf.keras.layers.Embedding layer to learn the spacial location of a dense vector of size 128 for each word in a 20K dictionary in the context of all the reviews. The embedding would minimize the euclidean distance of the similar words in 128 dimensions. The embedding layer initializes the weights and through the back propagation from the LSTM layer the weights learn the context. The embedding layer is a blueprint that provides 128 output dimensions based on 20K dictionary size (features) in input sequences with time steps =80.
LSTM layer puts the words into the context as a sequence of time steps = 80. Based on the training label it learns the relationship of the words and back propagates the loss gradient to train the weights through the entire network.

### Padding the sequence
To use the lstm pad the sequences to a fixed length (step size)
```
from tensorflow.keras.preprocessing import sequence
x_train = sequence.pad_sequences(x_train, maxlen=timesteps)
x_test = sequence.pad_sequences(x_test, maxlen=timesteps)
print('x_train shape: ', x_train.shape)
print('x_test shape: ', x_test.shape)
```
x_train shape:  (25000, 80)
x_test shape:  (25000, 80)

### Use the NN to learn a word embedding in the context
A word embedding is a class of approaches for representing words and documents using a dense vector representation. It is an improvement over the traditional bag-of-word model encoding schemes where large sparse vectors were used to represent each word. 

Instead, in an embedding, words are represented by dense vectors where a vector represents the projection of the word into a continuous vector space. The position of a word within the vector space is learned from text and is based on the words that surround the word when it is used. The position of a word in the learned vector space is referred to as its embedding. Two popular examples of methods of learning word embeddings from text include: Word2Vec. GloVe. In addition to these carefully designed methods, a word embedding can be learned as part of a deep learning model. This can be a slower approach, but tailors the model to a specific training dataset.

**Kera embedding layer:** Keras offers an Embedding layer that can be used for neural networks on text data. It requires that the input data be integer encoded, so that each word is represented by a unique integer. This data preparation step can be performed using the Tokenizer API also provided with Keras. The Embedding layer is initialized with random weights and will learn an embedding for all of the words in the training dataset.
```
model = Sequential()
model.add(Embedding(input_dim= num_features, output_dim = 128, input_length=timesteps))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
```
### LSTM Text generation with Keras
We use 137 articles to learn how to generate text based on the semantic learned.

### Steps to generate Text
* Import articles. (`136` articles in 136 rows)
* create one giant text of all the articles >> `text`
* create a bag of unique characters with a corresponding unique integer >> `char, char_int, int_char`
* create a list of interleaved sequences of the same length characters, `40` represented with integer keys of the characters as input data and a list of corresponding integer represented next_character. (the character after the sequence) as the target label >> `sequences, next_char`.
interleaved step sizes for generating sequences are 5 characters. smaller number is similar to lowering the learning rate. increasing the length of the sequence is similar to train a dataset with larger batch size.
  > we can't train on this data as the next character *prediction* is a floating point that if it's rounded up or down, does not necessarily point to a predicted character as the integer representation of characters are random not ordinal
* create multi dimensional boolean array for X: 
  1. axis0: sequence number >> size `178374`
  2. axis1: position of the character in sequence >> size `40` this contains timestep information
  3. axis2: identifier of the character in the bag of characters >> size `33` this is the feature dimension for characters
* create a multi dimensional boolean array for Y:
  1. axis0: sequence number >> size `178374`
  2. axis1: identifier of the next_char in the bag of characters >> size `33`
* Build the LSTM model
* define a callback at the end of each epoch:
  1. pick a random prompt (index) in the concatenated giant `text`
  2. grab the 40 characters of the `text` as the query seed for character generation
  3. convert the query seed to X.shape, i.e. (1, 40, 33) >> `x_pred`
  4. get a y prediction from model after each epoch training, shape: (1, 33) >> `preds` an array of 33 floating values between 0 and 1, 1 being the strongest possibility for being the next_char
  5. scale the values of y array to proba and take one *draw* from the array considering the value of proba. grab the selected char from the draw and print it as the next char after the sequence.
  6. shift the input sequence to the right by 1 and predict the next char again >> get `400` characters iteratively
* This way we work with probability of all the characters as the next char instead of a floating number that is supposed to resemble one of the characters integer representation.

### Input to the keras layer
The `input_shape` parameter simply tells the input layer what the shape of one sample looks like.
If one sample of the input tensor only has one dimension – which is the case with one-dimensional / flattened arrays, in this case, you can also simply use `input_dim`: specifying the number of elements within that first dimension only.

### LSTM input sequence
We have an input sequence of characters. So our sequence step size is 40. each characater has 33 features (that all except one has False value) and we have total of 178374 sequences. So the LSTM tensor input has 40 time steps and each step has 33 features. We would also provide 128 parallel lstm hidden units in each LSTM cell that is connected to each time step.
```
model = Sequential()
model.add(LSTM(128, input_shape=(timesteps, len(chars))))
model.add(Dense(units=len(chars), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
```

## ann_rnn_lstm-431a.ipynb:
The purpose of this notebook is to regenerate some text from Shakespeare writings. After cleaning the writings and removing the odd characters and extra white spaces, we create a one giant text from all his work and save it as `text` variable. Our sequences are comprised of characters. The length of our character set is 36, each of them assigned to an ordinal integer. The characters are saved into variable `char` and the mapping dictionaries are saved as `char_int` and `int_char`.
### Generate training data sequences and target labels:
We capture 40 consecutive characters as our time step with stepping=20. The stepping dictates how far apart to move the start index to capture the sequence in the text` variable. For generating text we don’t have any test data and the entire dataset is used to train the NN. This leads to approximately 700K sequences. The target label is the character after the last character in a sequence.
### LSTM as Input Layer:
We can either use LSTM or Embedding as the input layer. Here each character is going to be considered in terms of its relationship with other characters. If we use LSTM as the input layer, the input tensor has a shape of (batch_size, timestep, features). This is defined by `input_shape` or `input_dim` parameters. Batch_size can be skipped and is taken as None which indicates that it’s dynamically determined. The same is true for timestep. Either we can skip it and let’s be entered and None, so that the timestep can be detected dynamically or we explicitly enter the value. In the latter case despite training the model with a fixed timestep we can still pass an input query with a different time step and the model will process that. Here are different ways we can pass the input tensor information to the input layer. Features refers to the size of the dictionary. Here we have a dictionary of characters with a length of 36 and the time step is 40 characters.
With LSTM as the input layer we need to specify:
* `input_shape=[timesteps, features]` or
* `input_shape = [None, features]` or
* `input_dim = features`
> In prediction of all the above three cases, the sequence could have bigger or smaller step size and it would still work.
```
# input method 1
model_lstm.add(LSTM(128, input_shape=(timestep, dict_size)))
 
# input method 2
model_lstm.add(LSTM(128, input_shape=(None, dict_size)))
 
# input method 3
model_lstm.add(LSTM(128, input_dim=dict_size))
```
### Reshape the input tensor:
To match the definition we need to convert the input tensor sequences into a three dimension tensor. The first axis is the sample number with 1 feature. The second axis is the timing step and it has 40 features indicating the position of the character in the sequence. The third axis is the input features or dictionary size which has 36 features. Therefore, `X.shape=(1,timingstep=40, features=len(chars)=36)`.
 
### Embedding as Input layer:
With Embedding as the input layer, we can specify the input tensor shape in different ways. `input_dim` in the Embedding layer refers to features or the size of the dictionary (36). A parameter named `input_length` can arbitrarily be used to specify the timing step (40) in character sequence, or be left None to be handled dynamically.
1. input_dim=features or
2. input_dim=features, input_length=stepsize or
3. input_dim=features, input_shape=(stepsize,)
> In prediction of all three cases, the sequence could have bigger or smaller step size and it would still work with some warning.
```
# define input layer
# method1, timestep will be derived dynamically
model_embed.add(Embedding(output_dim=64, input_dim=dict_size))
 
# method2, timestep of the sequence is fixed:
# The query input still can be in different timesteps and we would only get warning
model_embed.add(Embedding(input_dim=dict_size, output_dim = 64, input_length=timestep))
 
# method3, use input_shape to define timestep
model_embed.add(Embedding(output_dim=64, input_dim=dict_size, input_shape=(timestep,)))
```
Note that here embedding is used for the characters not the words. We write a callback function to generate 400 characters after receiving a random see at the end of each epoch. As epochs progress and back propagation adjusts the parameters we get more improvement in generated text based on the random seeds.


### Libraries:
```
from __future__ import print_function
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Bidirectional
from tensorflow.keras.layers import LSTM
from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.optimizers import RMSprop
import random
import sys
import os
import re
```
