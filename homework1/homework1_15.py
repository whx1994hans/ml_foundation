import numpy as np


def load_train_data(file_name):
    training_set = np.loadtxt(file_name)
    x0 = np.ones(training_set.shape[0])
    training_set = np.insert(training_set, 0, x0, axis=1)
    return training_set


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


def pla(features, labels, w):
    update_times = 0
    error = False
    while error or ((w == 0).all()):
        error = False
        for feature, label in zip(features, labels):
            predict_y = pla_hypothesis_value(w, feature)
            if predict_y == label:
                continue
            else:
                w = w + label * feature
                update_times += 1
                error = True

    return update_times, w


file_name = 'hw1_15_train.txt'
w = np.zeros(5)
training_set = load_train_data(file_name)
train_feature, train_label = split_features_and_labels(training_set)
update_times, w = pla(train_feature, train_label, w)

print(update_times)
print(w)

print(train_feature[1, :])
print(train_label[1])
