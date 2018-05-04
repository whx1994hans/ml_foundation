import numpy as np


def load_data(file_name):
    dataset = np.loadtxt(file_name)
    x0 = np.ones(dataset.shape[0])
    dataset = np.insert(dataset, 0, x0, axis=1)
    return dataset


def split_features_and_labels(dataset):
    features = dataset[:, :-1]
    labels = dataset[:, dataset.shape[1]-1]
    return features, labels


def pla_hypothesis_value(w, x):
    predict_y = w.dot(x)

    if predict_y > 0:
        return 1.0
    else:
        return -1.0


def count_mistake(features, labels, w):
    mistake = 0

    for feature, label in zip(features, labels):
        y_predict = pla_hypothesis_value(w, feature)
        if y_predict != label:
            mistake += 1

    return mistake


def pla(features, labels, w):
    update_times = 0
    update_times_max = 50

    while update_times < update_times_max:
        for feature, label in zip(features,labels):
            predict_y = pla_hypothesis_value(w, feature)
            if predict_y == label:
                continue
            else:
                w = w + label * feature
                update_times += 1

    return w


def error_rate(test_set, w):
    test_features, test_labels = split_features_and_labels(test_set)
    mistake_nums = count_mistake(test_features, test_labels, w)
    error = mistake_nums / test_features.shape[0]
    return error


def random_cycles_pocket_n_times(train_file, test_file, n_times):
    training_set = load_data(train_file)
    test_set = load_data(test_file)
    error = 0

    for i in range(0, n_times):
        w = np.zeros(training_set.shape[1]-1)
        np.random.shuffle(training_set)
        features, labels = split_features_and_labels(training_set)
        w_pla = pla(features, labels, w)
        error += error_rate(test_set, w_pla)

    return error / n_times


train_file = 'hw1_18_train.txt'
test_file = 'hw1_18_test.txt'
avg_error = random_cycles_pocket_n_times(train_file, test_file, 2000)
print(avg_error)
