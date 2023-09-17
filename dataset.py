import matplotlib.image as mpimg

import os
import csv


def load_cifar_train(root):
    """
    takes root folder (where trainLabels.csv, train folder, and test folder are stored)
    returns list of tuples [(label, image_in_np_array), ....]
    """
    dataset = []
    train_path = os.path.join(root, "train")
    with open(os.path.join(root, "trainLabels.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            dataset.append(
                (
                    label2num(row[1]),
                    mpimg.imread(os.path.join(train_path, f"{row[0]}.png")).transpose(
                        2, 0, 1
                    ),
                )
            )
    return dataset


def label2num(label):
    dicc = {
        "airplane": 0,
        "automobile": 1,
        "bird": 2,
        "cat": 3,
        "deer": 4,
        "dog": 5,
        "frog": 6,
        "horse": 7,
        "ship": 8,
        "truck": 9,
    }
    return dicc[label]
