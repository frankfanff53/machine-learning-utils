import numpy as np


def confusion_matrix_init(y_gold: np.ndarray, y_prediction: np.ndarray, class_labels: np.ndarray = None) -> np.ndarray:
    """ Compute the confusion matrix.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        class_labels (np.ndarray): a list of unique class labels.
                               Defaults to the union of y_gold and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes.
                   Rows are ground truth per class, columns are predictions
    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros(
        (len(class_labels), len(class_labels)), dtype=int)

    for i, actual in enumerate(class_labels):
        for j, predicted in enumerate(class_labels):
            # Find the number of predictions with label pair (actual, predicted)
            for gold, prediction in zip(y_gold, y_prediction):
                if gold == actual and predicted == prediction:
                    confusion[i, j] += 1

    return confusion


def accuracy(y_gold: np.ndarray, y_prediction: np.ndarray) -> float:
    """ Compute the accuracy given the ground truth and predictions

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        float : the accuracy
    """

    assert len(y_gold) == len(y_prediction)

    try:
        return np.sum(y_gold == y_prediction) / len(y_gold)
    except ZeroDivisionError:
        return 0.


def accuracy_from_confusion(confusion: np.ndarray) -> float:
    """ Compute the accuracy given the confusion matrix

    Args:
        confusion (np.ndarray): shape (C, C), where C is the number of classes. 
                    Rows are ground truth per class, columns are predictions

    Returns:
        float : the accuracy
    """
    try:
        return np.sum(np.diag(confusion)) / np.sum(confusion)
    except ZeroDivisionError:
        return 0.


def precision(y_gold: np.ndarray, y_prediction: np.ndarray) -> float:
    """ Compute the precision score per class given the ground truth and predictions

    Also return the macro-averaged precision across classes.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (precisions, macro_precision) where
            - precisions is a np.ndarray of shape (C,), where each element is the 
              precision for class c
            - macro-precision is macro-averaged precision (a float) 
    """

    confusion = confusion_matrix_init(y_gold, y_prediction)

    # Compute the precision per class
    precisions = np.array([confusion[i, i] / np.sum(confusion[:, i])
                          for i in range(len(confusion))])

    # Compute the macro-averaged precision
    macro_p = np.average(precisions)

    return (precisions, macro_p)


def recall(y_gold: np.ndarray, y_prediction: np.ndarray) -> float:
    """ Compute the recall score per class given the ground truth and predictions

    Also return the macro-averaged recall across classes.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (recalls, macro_recall) where
            - recalls is a np.ndarray of shape (C,), where each element is the 
                recall for class c
            - macro-recall is macro-averaged recall (a float) 
    """

    confusion = confusion_matrix_init(y_gold, y_prediction)

    # Compute the recall per class
    recalls = np.array([confusion[i, i] / np.sum(confusion[i])
                       for i in range(len(confusion))])

    # Compute the macro-averaged recall
    macro_r = np.average(recalls)

    return (recalls, macro_r)


def f1_score(y_gold: np.ndarray, y_prediction: np.ndarray) -> float:
    """ Compute the F1-score per class given the ground truth and predictions

    Also return the macro-averaged F1-score across classes.

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where
            - f1s is a np.ndarray of shape (C,), where each element is the
              f1-score for class c
            - macro-f1 is macro-averaged f1-score (a float)
    """

    precisions, _ = precision(y_gold, y_prediction)
    recalls, _ = recall(y_gold, y_prediction)

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    # Compute the per-class F1
    sum_pr = precisions + recalls
    prod_pr = precisions * recalls
    f1_scores = 2 * prod_pr / sum_pr

    # Compute the macro-averaged F1
    macro_f = np.average(f1_scores)

    return (f1_scores, macro_f)
