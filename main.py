from sys import argv
import os
from cross_validation import CrossValidation
from knn import KNN
from metrics import accuracy_score
from normalization import *


def load_data():
    """
    Loads data from path in first argument
    :return: returns data as list of Point
    """
    if len(argv) < 2:
        print('Not enough arguments provided. Please provide the path to the input file')
        exit(1)
    input_path = argv[1]

    if not os.path.exists(input_path):
        print('Input file does not exist')
        exit(1)

    points = []
    with open(input_path, 'r') as f:
        for index, row in enumerate(f.readlines()):
            row = row.strip()
            values = row.split(',')
            points.append(Point(str(index), values[:-1], values[-1]))
    return points


def quest_3(points, k, n_fold):
    # this function prints the accuracy of every fold of a N-fold-cross-validation for N in n_fold.
    m = KNN(k)
    m.train(points)
    print("Question 3:")
    print("K={}".format(k))
    for fold in n_fold:
        print("{}-fold-cross-validation:".format(fold))
        cv = CrossValidation()
        cv.run_cv(points, fold, m, accuracy_score)


def quest_4(points):
    # this function prints the accuracy of 4 diffrent normalizers using 2 fold-cross-validation.
    print("Question 4:")
    for i in {5, 7}:
        print("K={}".format(i))
        for normalizer in [DummyNormalizer,SumNormalizer, MinMaxNormalizer, ZNormalizer]:
            m = KNN(i)
            n = normalizer()
            n.fit(points)
            cv = CrossValidation()
            average_score = cv.run_cv(n.transform(points), 2, m, accuracy_score)
            print("Accuracy of {} is {:.2f}".format(normalizer.__name__, average_score))
            print()


def run_knn(points):
    m = KNN(5)
    m.train(points)
    print(f'predicted class: {m.predict(points[0])}')
    print(f'true class: {points[0].label}')
    cv = CrossValidation()
    cv.run_cv(points, 10, m, accuracy_score)


if __name__ == '__main__':
    loaded_points = load_data()
    quest_3(loaded_points, 19, [2, 10, 20])
    quest_4(loaded_points)
