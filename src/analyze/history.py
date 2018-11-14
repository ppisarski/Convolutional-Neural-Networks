import matplotlib.pyplot as plt


def plot_training_history(history, filename='history.png'):
    """
    This function plots the training history.
    """
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(x, acc, 'orange', label='Training acc', alpha=0.9)
    ax1.plot(x, val_acc, 'dodgerblue', label='Validation acc', alpha=0.9)
    ax2.plot(x, loss, 'orange', label='Training loss', alpha=0.9)
    ax2.plot(x, val_loss, 'dodgerblue', label='Validation loss', alpha=0.9)
    ax1.set_title('Training and validation accuracy')
    ax2.set_title('Training and validation loss')
    ax1.legend()
    ax2.legend()
    fig.savefig(filename)


def plot_training_histories(history, filename='history.png'):
    """
    This function plots the training history.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot([], [], 'orange', label='Training acc', alpha=0.9)
    ax1.plot([], [], 'dodgerblue', label='Validation acc', alpha=0.9)
    ax2.plot([], [], 'orange', label='Training loss', alpha=0.9)
    ax2.plot([], [], 'dodgerblue', label='Validation loss', alpha=0.9)
    ax1.set_title('Training and validation accuracy')
    ax2.set_title('Training and validation loss')
    ax1.legend()
    ax2.legend()
    for i in range(len(history)):
        acc = history[i].history['acc']
        val_acc = history[i].history['val_acc']
        loss = history[i].history['loss']
        val_loss = history[i].history['val_loss']
        x = range(1, len(acc) + 1)

        ax1.plot(x, acc, 'orange', alpha=0.9)
        ax1.plot(x, val_acc, 'dodgerblue', alpha=0.9)
        ax2.plot(x, loss, 'orange', alpha=0.9)
        ax2.plot(x, val_loss, 'dodgerblue', alpha=0.9)
    fig.savefig(filename)


def main():
    class Object(object):
        pass

    hist = [Object(), Object()]
    hist[0].history = {'acc': [0, 1, 4, 3, 4], 'val_acc': [0, 1, 2, 3, 4],
                       'loss': [0, 1, 2, 3, 4], 'val_loss': [0, 1, 2, 3, 4]}
    hist[1].history = {'acc': [0, 1, 2, 3, 4], 'val_acc': [0, 1, 2, 3, 4],
                       'loss': [0, 1, 2, 3, 4], 'val_loss': [0, 1, 2, 3, 4]}
    plot_training_history(hist[0])


if __name__ == '__main__':
    main()
