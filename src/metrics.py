import numpy as np
from typing import List, Set, Iterable
from collections import namedtuple


Evaluation = namedtuple('Evaluation', ['recall', 'lift'])
ExperimentResult = namedtuple('ExperimentResult', ['train', 'test'])


def _recall(top_predictions: Set[int], test_ones: List[int]) -> float:
    """
    Helper function used to calculate recall per user
    :param top_predictions: set of indices of predicted items for a given user
    :param test_ones: list of indices of ground truth items known to interact with a given user
    :return: recall: probability that a ground truth interaction will be predicted
    """
    if not test_ones:  # no ground truth examples provided - return NaN
        return np.nan
    ones_in_top = len(top_predictions.intersection(test_ones))
    recall = ones_in_top / len(test_ones)
    return recall


def recall_lift(
        all_predictions: np.array, train_ones: List[List[int]],
        test_ones: List[List[int]], m: int or Iterable[int]) -> Evaluation or List[Evaluation]:
    """
    Calculates average recall and lift per user given that items with top m scores are considered
    predictions of positive interactions
    :param all_predictions: full matrix of predictions of shape (num_users,  num_items)
    :param train_ones: training pairs to exclude from evaluation. The same format as in
        models.BlockCoordinateAscent
    :param test_ones: test pairs to evaluate
    :param m: number of top scores to use as predictions
    :return: recall, lift
    """
    predictions = all_predictions.copy()  # may cause memory issues for large datasets, used here
    #  to avoid changing original
    for i, one in enumerate(train_ones):
        predictions[i, one] = - np.inf  # predictions for training ones are set to negative
        # infinity not to be considered as top scores in evaluation
    predictions = predictions.argsort()
    if isinstance(m, int):
        m = [m]
    results = Evaluation(recall=[], lift=[])
    for _m in m:
        recall = np.zeros(shape=(predictions.shape[0],))
        lift = np.zeros(shape=(predictions.shape[0],))
        for i in range(len(train_ones)):
            recall[i] = _recall(set(predictions[i, -_m:]), test_ones[i])
            valid_predictions = all_predictions.shape[1] - len(train_ones[i])
            lift[i] = valid_predictions * recall[i] / _m
        results.recall.append(float(np.nanmean(recall)))
        results.lift.append(float(np.nanmean(lift)))
    if len(results.recall) == 1:
        return Evaluation(recall=results.recall[0], lift=results.lift[0])
    return results
