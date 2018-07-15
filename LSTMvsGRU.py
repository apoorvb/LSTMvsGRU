import numpy as np
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences
from keras.datasets import imdb


(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz",
                                                      num_words=20000,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=113,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)

x_train[0]
y_train[0]


num_tokens  = [len(tokens) for tokens in x_train + x_test]

num_tokens = np.array(num_tokens)

np.mean(num_tokens), np.max(num_tokens)

max_tokens = np.mean(num_tokens) + 2*np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens

x_train = pad_sequences(x_train, maxlen=max_tokens,
                            padding= 'pre', truncating= 'pre' )

x_test = pad_sequences(x_test, maxlen=max_tokens,
                            padding= 'pre', truncating= 'pre' )


#GRU model with 3 layers: output_dim - 8
model = Sequential()
model.add(Embedding(input_dim=20000,
                    output_dim=8,
                    input_length=max_tokens))

model.add(GRU(units=16, return_sequences=True))
model.add(GRU(units=8, return_sequences=True))
model.add(GRU(units=4))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss= 'binary_crossentropy' ,
              optimizer= 'Adam',
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train,
          validation_split=0.05, epochs=3, batch_size=64)

score, acc = model.evaluate(x_test, y_test,
                            batch_size=64)
print(score, acc)

#LSTM model with 3 layers:output_dim - 8
modelL1 = Sequential()
modelL1.add(Embedding(input_dim=20000,
                    output_dim=8,
                    input_length=max_tokens))

modelL1.add(LSTM(units=16, return_sequences=True))
modelL1.add(LSTM(units=8, return_sequences=True))
modelL1.add(LSTM(units=4))
modelL1.add(Dense(1, activation='sigmoid'))

modelL1.compile(loss= 'binary_crossentropy' ,
              optimizer= 'Adam',
              metrics=['accuracy'])
modelL1.summary()

modelL1.fit(x_train, y_train,
          validation_split=0.05, epochs=3, batch_size=64)

score, acc = modelL1.evaluate(x_test, y_test,
                            batch_size=64)
print(score, acc)

#GRU model with 3 layers:output_dim - 128
model2 = Sequential()
model2.add(Embedding(input_dim=20000,
                    output_dim=128,
                    input_length=max_tokens))

model2.add(GRU(units=16, return_sequences=True))
model2.add(GRU(units=8, return_sequences=True))
model2.add(GRU(units=4))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss= 'binary_crossentropy' ,
              optimizer= 'Adam',
              metrics=['accuracy'])
model2.summary()

model2.fit(x_train, y_train,
          validation_split=0.05, epochs=3, batch_size=64)

score, acc = model2.evaluate(x_test, y_test,
                            batch_size=64)
print(score, acc)

#LSTM model with 3 layers:output_dim - 128
modelL2 = Sequential()
modelL2.add(Embedding(input_dim=20000,
                    output_dim=128,
                    input_length=max_tokens))

modelL2.add(LSTM(units=16, return_sequences=True))
modelL2.add(LSTM(units=8, return_sequences=True))
modelL2.add(LSTM(units=4))
modelL2.add(Dense(1, activation='sigmoid'))

modelL2.compile(loss= 'binary_crossentropy' ,
              optimizer= 'Adam',
              metrics=['accuracy'])
modelL2.summary()

modelL2.fit(x_train, y_train,
          validation_split=0.05, epochs=3, batch_size=64)

score, acc = modelL2.evaluate(x_test, y_test,
                            batch_size=64)
print(score, acc)






























































































































































































