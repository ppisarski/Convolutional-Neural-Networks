import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def plot_confusion_matrix(true_labels, pred_labels, filename='confusion_matrix.png', normalize=False):
    """
    This function plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # compute the confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    print(cm)

    fig, ax = plt.subplots()
    ax.imshow(cm, aspect='equal', cmap=plt.cm.Oranges)
    ax.set_xticks(np.arange(np.amin(true_labels), np.amax(true_labels) + 1))
    ax.set_yticks(np.arange(np.amin(pred_labels), np.amax(pred_labels) + 1))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    ax.set_title('Accuracy score = {:.5f}'.format(accuracy_score(true_labels, pred_labels)))
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.savefig(filename)


def main():
    true_labels = [0, 2, 3, 4, 1, 2]
    pred_labels = [2, 2, 3, 4, 0, 2]
    plot_confusion_matrix(true_labels, pred_labels)


if __name__ == '__main__':
    main()
