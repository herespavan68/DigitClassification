"""
Utility functions for digit classification
"""

import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


def load_and_plot_digits():
    """
    Load the digits dataset and visualize the first 4 training images.
    
    Returns:
        digits: The loaded digits dataset
    """
    digits = datasets.load_digits()
    
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, label in zip(axes, digits.images, digits.target):
        ax.set_axis_off()
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title("Training: %i" % label)
    
    return digits


def prepare_data(digits):
    """
    Flatten the images and return the flattened data.
    
    Args:
        digits: The digits dataset
        
    Returns:
        tuple: (flattened_data, target_labels)
    """
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    return data, digits.target


def train_classifier(X_train, y_train):
    """
    Train an SVM classifier on the training data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        clf: Trained SVM classifier
    """
    clf = svm.SVC(gamma=0.001)
    clf.fit(X_train, y_train)
    return clf


def plot_predictions(X_test, predicted):
    """
    Visualize the first 4 test samples and their predictions.
    
    Args:
        X_test: Test features
        predicted: Predicted labels
    """
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")


def generate_classification_report(clf, y_test, predicted):
    """
    Generate and print the classification report.
    
    Args:
        clf: Trained classifier
        y_test: Test labels
        predicted: Predicted labels
    """
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n"
    )


def plot_confusion_matrix(y_test, predicted):
    """
    Plot and print the confusion matrix.
    
    Args:
        y_test: Test labels
        predicted: Predicted labels
        
    Returns:
        disp: ConfusionMatrixDisplay object
    """
    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    return disp


def rebuild_report_from_confusion_matrix(disp):
    """
    Rebuild classification report from confusion matrix.
    
    Args:
        disp: ConfusionMatrixDisplay object
    """
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix
    
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]
    
    print(
        "Classification report rebuilt from confusion matrix:\n"
        f"{metrics.classification_report(y_true, y_pred)}\n"
    )
