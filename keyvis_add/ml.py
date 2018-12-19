import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Flatten, Reshape
from keras import regularizers

from plotly import offline as py
import plotly.graph_objs as go
from plotly import tools

py.init_notebook_mode()

# Loads the training and test data sets (ignoring class labels)
(x_train, _), (x_test, _) = mnist.load_data()

# Scales the training and test data to range between 0 and 1.
max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_test = x_test.astype('float32') / max_value

# Reshape
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# x_train.shape

# Autoencoder

input_dim = x_train.shape[1]
encoding_dim = 32
compression_factor = float(input_dim) / encoding_dim
print("Compression factor: %s" % compression_factor)

autoencoder = Sequential()
autoencoder.add(
    Dense(encoding_dim, input_shape=(input_dim,), activation='relu')
    )
autoencoder.add(
    Dense(input_dim, activation='sigmoid')
    )

autoencoder.summary()

input_img = Input(shape=(input_dim,))
encoder_layer = autoencoder.layers[0]
encoder = Model(input_img, encoder_layer(input_img))

encoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

num_images = 10
np.random.seed(42)
random_test_images = np.random.randint(x_test.shape[0], size=num_images)


encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

encoded_imgs[0]
decoded_imgs[0]

fig = tools.make_subplots(rows=1, cols=3, print_grid=False)

t1 = go.Heatmap(z=x_test[random_test_images[0]].reshape(28, 28), showscale=False)

fig.append_trace(t1, 1, 1)
# fig.append_trace(trace2, 1, 2)
# fig.append_trace(trace3, 1, 3)

for i in map(str,range(1, 4)):
        y = 'yaxis'+ i
        x = 'xaxis' + i
        fig['layout'][y].update(autorange='reversed',
                                showticklabels=False, ticks='', scaleanchor = 'x')
        fig['layout'][x].update(showticklabels=False, ticks='')

fig['layout'].update(height=600)
py.iplot(fig)
