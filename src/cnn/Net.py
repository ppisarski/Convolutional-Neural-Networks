import numpy as np

from keras.utils.generic_utils import to_list
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator


def create_model():
    """Create Neural Network model"""
    model = Sequential()
    model.add(Convolution2D(48, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Convolution2D(48, (5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    # Compile the model
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


class Net(object):
    def __init__(self):
        self.model = None
        self.batch_size = 64

    def fit(self, x, y, **kwargs):
        """Constructs a new model and fits the model to `(x, y)`.

        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`

        # Returns
            history : object
                details about the training history at each epoch.
        """
        x = x.values.astype('float32')
        x = x.reshape(x.shape[0], 28, 28, 1)

        # one hot encoding of labels
        y = to_categorical(y)

        if 'validation_data' in kwargs:
            vx = kwargs['validation_data'][0].values.astype('float32')
            vx = vx.reshape(vx.shape[0], 28, 28, 1)
            vy = to_categorical(kwargs['validation_data'][1])
            kwargs['validation_data'] = (vx, vy)

        self.model = create_model()

        # data augmentation to prevent overfitting
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        datagen.fit(x)

        # learning rate reduction
        annealer = ReduceLROnPlateau(monitor='val_acc',
                                     patience=3,
                                     verbose=1,
                                     factor=0.5,
                                     min_lr=0.00001)

        return self.model.fit_generator(datagen.flow(x, y, batch_size=self.batch_size),
                                        steps_per_epoch=x.shape[0] // self.batch_size,
                                        callbacks=[annealer], **kwargs)

    def predict(self, x, **kwargs):
        """Returns the class predictions for the given test data.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.

        # Returns
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """
        x = x.values.astype('float32')
        x = x.reshape(x.shape[0], 28, 28, 1)

        return self.model.predict_classes(x, **kwargs)

    def score(self, x, y, **kwargs):
        """Returns the mean accuracy on the given test data and labels.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.evaluate`.

        # Returns
            score: float
                Mean accuracy of predictions on `x` wrt. `y`.

        # Raises
            ValueError: If the underlying model isn't configured to
                compute accuracy. You should pass `metrics=["accuracy"]` to
                the `.compile()` method of the model.
        """
        x = x.values.astype('float32')
        x = x.reshape(x.shape[0], 28, 28, 1)

        y = to_categorical(y)

        outputs = self.model.evaluate(x, y, **kwargs)
        outputs = to_list(outputs)
        for name, output in zip(self.model.metrics_names, outputs):
            if name == 'acc':
                return output
        raise ValueError('The model is not configured to compute accuracy. '
                         'You should pass `metrics=["accuracy"]` to '
                         'the `model.compile()` method.')


class CommitteeNet(object):
    def __init__(self):
        self.n_members = 15
        self.model = [None] * self.n_members
        self.n_classes = 0
        self.batch_size = 64

    def fit(self, x, y, **kwargs):
        """Constructs a new model and fits the model to `(x, y)`.

        # Arguments
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`

        # Returns
            history : object
                details about the training history at each epoch.
        """
        x = x.values.astype('float32')
        x = x.reshape(x.shape[0], 28, 28, 1)

        # one hot encoding of labels
        y = to_categorical(y)
        self.n_classes = y.shape[1]

        if 'validation_data' in kwargs:
            vx = kwargs['validation_data'][0].values.astype('float32')
            vx = vx.reshape(vx.shape[0], 28, 28, 1)
            vy = to_categorical(kwargs['validation_data'][1])
            kwargs['validation_data'] = (vx, vy)

        for idx in range(self.n_members):  # instantiate committee of neural networks
            self.model[idx] = create_model()

        # data augmentation to prevent overfitting
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        datagen.fit(x)

        # learning rate reduction
        annealer = ReduceLROnPlateau(monitor='val_acc',
                                     patience=3,
                                     verbose=1,
                                     factor=0.5,
                                     min_lr=0.00001)

        history = [None] * self.n_members
        for idx in range(self.n_members):
            history[idx] = self.model[idx].fit_generator(datagen.flow(x, y, batch_size=self.batch_size),
                                                         steps_per_epoch=x.shape[0] // self.batch_size,
                                                         callbacks=[annealer], **kwargs)
            print("Net {:d}: Train accuracy={:.5f}, Validation accuracy={:.5f}".format(
                idx + 1, max(history[idx].history['acc']), max(history[idx].history['val_acc'])))
        return history

    def predict(self, x, **kwargs):
        """Returns the class predictions for the given test data.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            **kwargs: dictionary arguments
                Legal arguments are the arguments
                of `Sequential.predict_classes`.

        # Returns
            preds: array-like, shape `(n_samples,)`
                Class predictions.
        """
        x = x.values.astype('float32')
        x = x.reshape(x.shape[0], 28, 28, 1)

        classes = np.zeros((x.shape[0], self.n_classes))
        for idx in range(self.n_members):
            classes += to_categorical(self.model[idx].predict_classes(x, **kwargs))
        return np.argmax(classes, axis=1)

    def score(self, x, y, **kwargs):
        """Returns the mean accuracy on the given test data and labels.

        # Arguments
            x: array-like, shape `(n_samples, n_features)`
                Test samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.evaluate`.

        # Returns
            score: float
                Mean accuracy of predictions on `x` wrt. `y`.

        # Raises
            ValueError: If the underlying model isn't configured to
                compute accuracy. You should pass `metrics=["accuracy"]` to
                the `.compile()` method of the model.
        """
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(x))
