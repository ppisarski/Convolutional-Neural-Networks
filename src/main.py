import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from sklearn import svm

from analyze import plot_confusion_matrix


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--train',
                        dest='train', help='training and validation dataset',
                        metavar='TRAINDATA', required=True)
    parser.add_argument('--test',
                        dest='test', help='test dataset',
                        metavar='TESTDATA', required=True)
    parser.add_argument('--model',
                        dest='model', help='specify model',
                        metavar='MODEL', required=True,
                        choices=['SVC'])
    parser.add_argument('--testsize',
                        dest='testsize', help='size of testset for training/validation split',
                        metavar='TESTSIZE', default=0.1)
    return parser


def read_datasets(train_file, test_file):
    labeled_images = pd.read_csv(train_file)
    train_images = labeled_images.iloc[:, 1:]
    train_labels = labeled_images.iloc[:, :1]
    test_images = pd.read_csv(test_file)
    return train_images, train_labels, test_images


def write_labels(results, results_file='results.csv'):
    df = pd.DataFrame(results)
    df.index.name = 'ImageId'
    df.index += 1
    df.columns = ['Label']
    df.to_csv(results_file, header=True)


def main():
    parser = build_parser()
    options = parser.parse_args()

    train_data_images, train_data_labels, test_data_images = read_datasets(options.train, options.test)

    # simplify images by scaling them
    train_data_images /= 255
    test_data_images /= 255
    # or alternatively making them black and white
    # train_data_images[train_data_images > 0] = 1
    # test_data_images[test_data_images > 0] = 1

    # split training set into training and validation
    train_images, valid_images, train_labels, valid_labels = \
        train_test_split(train_data_images, train_data_labels, test_size=options.testsize, random_state=0)

    clf = None
    if options.model.upper() == 'SVC':
        clf = svm.SVC()
        cm_filename = 'images/svc.png'
        results_filename = 'results/svc.csv'

    clf.fit(train_images, train_labels.values.ravel())
    print(clf.score(valid_images, valid_labels))

    # Predict the values from the validation dataset
    pred_valid_labels = clf.predict(valid_images)
    plot_confusion_matrix(valid_labels.values, pred_valid_labels, cm_filename)

    # label the test images
    results = clf.predict(test_data_images)
    write_labels(results, results_filename)

    print('Done')


if __name__ == '__main__':
    main()
