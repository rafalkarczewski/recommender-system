import os
import errno

from functools import partial
from typing import Tuple, Iterable

import pandas as pd


def _save_table(table: pd.DataFrame, output_path: str) -> None:
    """
    Helper function that adds an additional constant column equal to 1 (for unification of API for
    both benchmark datasets) and saves the table to a .csv file
    """
    constant_col = pd.Series([1] * table.shape[0], index=table.index)
    aug_table = pd.concat([table, constant_col], axis=1)
    aug_table.to_csv(output_path, header=False, index=False)


def _create_dir(dir_name: str) -> None:
    """
    Helper function that creates a directory or does nothing if it already
    exists
    """
    try:
        os.makedirs(dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def _sample(user: pd.DataFrame, size: int or float, seed: int) -> pd.DataFrame:
    """
    Helper function that samples row from a pandas.DataFrame
    :param user: an element of grouping the full dataset by user - corresponding to single user
    :param size: size of rows to sample. Either an absolute value (int) or fraction of input (float)
    :param seed: random seed used for reproducibility
    :return: sampled_rows: sampled data frame
    """
    if 0 < size < 1:
        n_samples = round(size * len(user))
    elif isinstance(size, int):
        n_samples = size
    else:
        raise ValueError(
            'size must be either a numer from (0, 1) interval'
            'or an integer. Received: {}.'.format(size)
        )
    sampled_rows = user.sample(n=n_samples, random_state=seed)
    return sampled_rows


def train_test_split(
        table: pd.DataFrame, user_id: str, doc_id: str,
        train_ratio: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a train-test split
    :param user_id: name of column containing user id
    :param doc_id: name of column containing document id
    :param train_ratio: amount of training data per user.
    :param seed: For reproducibility
    :return: tuple of 2 pandas DataFrames: (table_train, table_test)
    """
    sampling_function = partial(_sample, size=train_ratio, seed=seed)
    table_train = table.groupby([user_id]).apply(sampling_function)
    if table_train.shape[0] == 0:  # No rows were sampled
        return table_train, table
    table_train['train'] = 1
    full_table = pd.merge(
        table, table_train, how='left',
        left_on=[user_id, doc_id], right_on=[user_id, doc_id]
    )
    full_table = full_table.fillna(0).astype(int)  # rows not matched must be from test dataset, so we fill missing values of train column with 0
    table_train = full_table[full_table.train == 1][[user_id, doc_id]]
    table_test = full_table[full_table.train == 0][[user_id, doc_id]]
    return table_train, table_test


def train_test_folds(
        data_path: str, output_dir: str, user_id: str, doc_id: str,
        train_size: int or float, train_sizes: Iterable[int],
        full_seed: int, partial_seed: int) -> None:
    """
    Creates a holdout test set and splits the remaining training set into
    cross-validation (train, test) pairs
    :param data_path: path to a full dataset with columns user_id and
        doc_id. Each entry indicates a positive training example
    :param output_dir: name of the directory where train and test files
        will be created
    :param user_id: name of the user column in the dataset
    :param doc_id: name of the item column in the dataset
    :param train_size: how many (int) or what percentage (float)
        of items per user to use as training set
    :param train_sizes: For each element: train_size, a pair
        (train_set, test_set) is created, where:
            train_set contains train_size documents per user
            test_set contains remaining records
    :param full_seed: Int; for reproducibility; to be used for initial
        test-holdout split
    :param partial_seed: Int; for reproducibility; to be used for further
       splits of the train set
    """
    table = pd.read_csv(data_path)[[user_id, doc_id]]
    full_table_train, full_table_test = train_test_split(
        table, user_id, doc_id, train_size, full_seed)
    _create_dir(output_dir)
    if full_table_test.shape[0] > 0:
        test_filepath = os.path.join(output_dir, 'hold_out.csv')
        _save_table(full_table_test, test_filepath)
    for train_size in train_sizes:
        table_train, table_test = train_test_split(
            full_table_train, user_id, doc_id, train_size, partial_seed
        )
        fold_name = 'training_size_' + str(train_size) + '.csv'
        train_name = 'train_' + fold_name
        valid_name = 'valid_' + fold_name
        _save_table(table_train, os.path.join(output_dir, train_name))
        _save_table(table_test, os.path.join(output_dir, valid_name))


if __name__ == "__main__":

    train_test_folds(
        data_path='../data/user-info.csv', output_dir='../data/folds_no_test',
        user_id='user.id', doc_id='doc.id', train_size=1-1e-6,
        train_sizes=range(1, 6), full_seed=42, partial_seed=42)
