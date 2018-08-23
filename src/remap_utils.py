import os
import re
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import List, Generator, Union, Dict, Iterable
from models import BlockCoordinateAscent
from metrics import recall_lift, ExperimentResult
from collections import namedtuple
import scipy
from functools import reduce
from sklearn.model_selection import KFold


OnesList = List[List[int]]
Experiment = namedtuple('Experiment', ['name', 'train_ones', 'test_ones'])
CrossValidationMetric = namedtuple('CrossValidationMetric', ['mean', 'sem'])


def _file_to_list(filename: str, n_users: int) -> OnesList:
    """
    Helper function reading a csv file and trasforming it into an appropriate format.
    A file contains 3 columns: user_id, item_id, 1 (constant value). This function
    transforms it into a list, where i-th entry is a list of ids of items that
    appeared together with i-th user in the file
    """
    ones_list = [[] for _ in range(n_users)]
    df = pd.read_csv(filename, header=None)
    for row_id, (user_id, item_id, _) in df.iterrows():
        ones_list[user_id - 1].append(
            item_id - 1)  # ids of both users and items start from 1 instead of 0
    return ones_list


def _train_test_pairs(directory: str, n_users: int) -> Generator[Experiment, None, None]:
    """
    Yields all train-test pairs of csv files in the specified directory
    """
    valid_files = [filename for filename in os.listdir(directory) if filename.endswith('.csv')]
    train_files = [filename for filename in valid_files if filename.startswith('train')]
    for train_file in train_files:
        test_file = train_file.replace('train', 'test')
        split_ind = re.match('train_?', train_file).end()
        experiment_name = train_file[split_ind:-4]
        train_ones = _file_to_list(os.path.join(directory, train_file), n_users)
        test_ones = _file_to_list(os.path.join(directory, test_file), n_users)
        yield Experiment(name=experiment_name, train_ones=train_ones, test_ones=test_ones)


def _transpose_ones(ones_list: OnesList, n_items: int) -> OnesList:
    """
    Helper function that modifies the list of positive examples to swap the meaning of user and
    item. This is used in the BlockCoordinateAscent model, where we assume that we provide the
    embeddings of items(proteins), whereas in the REMAP setting we provide the embeddings of
    chemicals(users).
    """
    transposed_ones = [[] for _ in range(n_items)]
    for user_id, item_ids in enumerate(ones_list):
        for item_id in item_ids:
            transposed_ones[item_id].append(user_id)
    return transposed_ones


def evaluate(
        embeddings: np.ndarray, train_ones: OnesList, test_ones: OnesList,
        n_epochs: int, n_items: int, m: int or Iterable[int]) -> ExperimentResult:
    """
    Function that builds a model and computes metrics on the train and test sets
    """
    transposed_ones = _transpose_ones(train_ones, n_items)
    model = BlockCoordinateAscent(transposed_ones, embeddings.copy())
    model.fit(n_epochs=n_epochs)
    predictions = model.predict().T  # we transpose the output to swap the users with items back to
    # the original state
    train_results = recall_lift(predictions, [[]] * len(train_ones), train_ones, m)
    test_results = recall_lift(predictions, train_ones, test_ones, m)
    return ExperimentResult(train=train_results, test=test_results)


def _flatten(ones_list: OnesList) -> np.ndarray:
    """
    Helper function that transforms the list of ones into a numpy array with two columns: user_id,
    item_id. Necessary for efficient sampling of (user, item) pairs
    """
    flat_list = []
    for i, ones in enumerate(ones_list):
        for one in ones:
            flat_list.append((i, one))
    return np.array(flat_list)


def _inflate(flat_list: np.ndarray, n_users: int) -> OnesList:
    """
    Inverse function of _flatten. It takes an array with two columns: user_id, item_id and
    transforms it into a OnesList, i.e. a list, where i-th element is a list of ids of items which
    appeared in one row with i-th user
    """
    inflated_list = [[] for _ in range(n_users)]
    for user_id, item_id in flat_list:
        inflated_list[user_id].append(item_id)
    return inflated_list


def _merge(*ones_lists: Union[OnesList]) -> OnesList:
    """
    Merges OnesLists into one OnesList
    """
    assert all([type(ones_list) == list for ones_list in ones_lists])
    assert len(set([len(ones_list) for ones_list in ones_lists])) == 1
    merged_ones = [None for _ in range(len(ones_lists[0]))]
    for i, user_ones in enumerate(zip(*ones_lists)):
        merged_ones[i] = reduce(lambda x, y: x + y, list(user_ones))
    return merged_ones


def cross_validation(train_ones: OnesList, test_ones: OnesList, k: int):
    """
    Implements a  cross validation scheme used in "Large-Scale Off-Target Identification Using Fast
    and Accurate Dual Regularized One Class Collaborative Filtering and Its Application to Drug
    Repurposing".
    It splits the test set into k folds and during experiments each of the folds is used for
    evaluation and the remainder (k - 1 folds + full training set) are used for training
    """
    if k == 1:
        yield Experiment(name='fold_0', train_ones=train_ones, test_ones=test_ones)
        return None
    kf = KFold(n_splits=k)
    flat_test_ones = _flatten(test_ones)
    n_users = len(train_ones)
    for fold_id, (train_index, test_index) in enumerate(kf.split(flat_test_ones)):
        train_fold, test_fold = flat_test_ones[train_index], flat_test_ones[test_index]
        train_fold = _inflate(train_fold, n_users=n_users)
        train_fold = _merge(train_ones, train_fold)  # include the full train set in the train fold
        test_fold = _inflate(test_fold, n_users=n_users)
        yield Experiment(name='fold_' + str(fold_id), train_ones=train_fold, test_ones=test_fold)


def _validate_ones(ones_list: OnesList) -> None:
    assert type(ones_list) == list
    assert all(len(set(ones)) == len(ones) for ones in ones_list)  # no duplicates


def _validate_split(train_ones_list, test_ones_list):
    _validate_ones(train_ones_list)
    _validate_ones(test_ones_list)
    assert type(train_ones_list) == type(test_ones_list) == list
    assert len(train_ones_list) == len(test_ones_list)  # the same number of users
    assert all([
        not set(train_ones).intersection(set(test_ones))
        for train_ones, test_ones in zip(train_ones_list, test_ones_list)
    ])  # ensure no data leakage


def _gather_results(cv_results: List[ExperimentResult]) -> Dict[str, CrossValidationMetric]:
    """
    Function transforming list of results per crossvalidation fold and calculating the mean and the
    standard error of the mean for all the metrics and for both train and test folds
    """
    output = {}
    for eval_set in ['train', 'test']:
        for metric in ['recall', 'lift']:
            results = [getattr(getattr(result, eval_set), metric) for result in cv_results]
            result = CrossValidationMetric(
                mean=np.mean(results, axis=0),
                sem=scipy.stats.sem(results, axis=0)
            )
            output[eval_set + '_' + metric] = result
    return output


def evaluate_dir(
        directory: str, users_embedding: np.ndarray, n_items: int, epochs: int,
        m: int, folds: int) -> Dict[str, Dict[str, CrossValidationMetric]]:
    """
    Iterates over all train test pairs in the directory and performs cross validation for each pair
    reporting the mean and the standard error of the mean for recall and lift for both train and
    test folds
    :param directory: Directory containing train-test pairs of .csv files (with consistent names,
    e.g. train_N2L1to5.csv and test_N2L1to5.csv)
    :param users_embedding: vectorized representations of users
    :param m: Number of items to predict for each user; used for metrics
    """
    results = {}
    n_users = users_embedding.shape[0]
    for experiment_name, train_ones, test_ones in _train_test_pairs(directory, n_users):
        print('Experiment:', experiment_name)
        experiment_results = []
        for fold in cross_validation(train_ones, test_ones, folds):
            print(fold.name)
            k_fold_train = fold.train_ones
            k_fold_test = fold.test_ones
            _validate_split(k_fold_train, k_fold_test)
            fold_results = evaluate(
                embeddings=users_embedding, train_ones=k_fold_train, test_ones=k_fold_test,
                n_epochs=epochs, n_items=n_items, m=m)
            experiment_results.append(fold_results)
        results[experiment_name] = _gather_results(experiment_results)
    return results


def postprocess_results(
        results: Dict[str, Dict[str, CrossValidationMetric]],
        additional_column: str) -> pd.DataFrame:
    """
    Transforms the results dictionary into a digestible data frame
    :param results: output of evaluate_dir function - dictionary of results for all experiments
        including all metrics for both train and test sets
    :param additional_column: name of the additional column (other than "number of known targets"
        common for all experiments) that differentiates experiments. One of two values:
        "number of ligands per protein" or "chemical structural similarity"
    :return: Data frame with each row corresponding to one experiment reporting cross validated test
    recall and its standard error of the mean
    """
    output = defaultdict(list)
    for experiment_name, experiment_results in results.items():
        spilt_ind = re.match('N\d(up)?', experiment_name).end()
        known_targets = experiment_name[:spilt_ind]
        additional_info = experiment_name[spilt_ind:]
        output['known_targets'].append(known_targets)
        output[additional_column].append(additional_info)
        result_metrics = experiment_results['test_recall']
        output['TPR_mean'].append(result_metrics.mean)
        output['TPR_sem'].append(result_metrics.sem)
    return pd.DataFrame(output)
