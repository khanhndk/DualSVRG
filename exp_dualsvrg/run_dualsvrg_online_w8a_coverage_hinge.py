from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import pytest

from sklearn import metrics
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import load_svmlight_file

import numpy as np
import os

import multiprocessing as mp

from male.configs import random_seed
from male.datasets import demo
from male.common import data_dir
from male.models.kernel.dualsvrg_online_v2 import OnlineDualSVRG
from male.callbacks import Display

from male.utils.data_utils import get_file
from male.utils.data_utils import data_info


def test_sgd_visualization_2d(block_figure_on_end=False):
    (x_train, y_train), (_, _) = demo.load_synthetic_2d()

    predict_display = Display(
        freq=1,
        dpi='auto',
        block_on_end=block_figure_on_end,
        monitor=[{'metrics': ['predict'],
                  'title': "Visualization",
                  'xlabel': "X1",
                  'ylabel': "X2",
                  'grid_size': 100,
                  'marker_size': 10,
                  'left': None,
                  'right': None,
                  'top': None,
                  'bottom': None
                  }]
    )

    display = Display(
        freq=1,
        dpi='auto',
        block_on_end=block_figure_on_end,
        monitor=[{'metrics': ['mistake_rate'],
                  'type': 'line',
                  'title': "Mistake Rate",
                  'xlabel': "data points",
                  'ylabel': "Error",
                  }]
    )

    learner = OnlineDualSVRG(
        regular_param=0.01,
        learning_rate_scale=1.0,
        gamma=10,
        rf_dim=400,
        cache_size=6,
        freq_update_full_model=10,
        oracle='coverage',
        core_max=10,
        coverage_radius=0.5,
        loss_func='hinge',
        smooth_hinge_theta=0.5,
        smooth_hinge_tau=0.5,
        callbacks=[display],
        metrics=['mistake_rate'],
        random_state=random_seed())

    learner.fit(x_train, y_train)
    y_train_pred = learner.predict(x_train)
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))


def run_gridsearch_single_core():
    print("========== Tune parameters for IGKOL for classification ==========")

    data_name = 'svmguide1'
    n_features = 4

    train_file_name = os.path.join(data_dir(), data_name + '_train.libsvm')
    test_file_name = os.path.join(data_dir(), data_name + '_test.libsvm')

    print(train_file_name)
    print(test_file_name)

    if not os.path.exists(train_file_name):
        raise Exception('File not found')
    if not os.path.exists(test_file_name):
        raise Exception('File not found')

    x_train, y_train = load_svmlight_file(train_file_name, n_features=n_features)
    x_test, y_test = load_svmlight_file(test_file_name, n_features=n_features)

    x_train = x_train.toarray()
    x_test = x_test.toarray()

    x_total = np.vstack((x_train, x_test))
    y_total = np.concatenate((y_train, y_test))

    print("Number of total samples = {}".format(x_total.shape[0]))

    params = {'regular_param': [0.0001, 0.00001],
              'gamma': [0.25, 0.5, 1, 2]}

    candidate_params_lst = list(ParameterGrid(params))
    mistake_rate_lst = []
    run_param_lst = []
    for candidate_params in candidate_params_lst:
        clf = OnlineDualSVRG(
            # regular_param=0.01,
            learning_rate_scale=0.8,
            # gamma=2.0,
            rf_dim=400,
            num_epochs=1,
            freq_update_full_model=100,
            oracle='coverage',
            core_max=100,
            coverage_radius=0.9,
            loss_func='logistic',
            smooth_hinge_theta=0.5,
            smooth_hinge_tau=0.5,
            random_state=3333,
            **candidate_params,
        )
        clf.fit(x_total, y_total)
        print('Params:', candidate_params, 'Mistake rate:', clf.mistake_rate)

        mistake_rate_lst.append(clf.mistake_rate)
        run_param_lst.append(candidate_params)

    idx_best = np.argmin(np.array(mistake_rate_lst))
    print('Best mistake rate: {}'.format(mistake_rate_lst[idx_best]))
    print('Best params: {}'.format(run_param_lst[idx_best]))


loss_func = 'hinge'
oracle = 'coverage'
dataset = 'w8a'


def run_one_candidate(candidate_params):
    np.random.seed(random_seed())
    mistake_rate_avg = 0
    train_time_avg = 0
    num_runs = 3
    for ri in range(num_runs):
        print('----------------------------------')
        print('Run #{0}:'.format(ri + 1))

        # np.random.seed(5555)
        # np.random.seed(4444)
        data_name = dataset
        n_features = 300

        train_file_name = os.path.join(data_dir(), data_name + '_train.libsvm')
        test_file_name = os.path.join(data_dir(), data_name + '_test.libsvm')

        if not os.path.exists(train_file_name):
            raise Exception('File not found')
        if not os.path.exists(test_file_name):
            raise Exception('File not found')

        x_train, y_train = load_svmlight_file(train_file_name, n_features=n_features)
        x_test, y_test = load_svmlight_file(test_file_name, n_features=n_features)

        x_train = x_train.toarray()
        x_test = x_test.toarray()

        x_total = np.vstack((x_train, x_test))
        y_total = np.concatenate((y_train, y_test))

        print('Num total samples: {}'.format(x_total.shape[0]))

        clf = OnlineDualSVRG(
            # regular_param=0.01,
            learning_rate_scale=0.8,
            # gamma=2.0,
            rf_dim=4000,
            num_epochs=1,
            freq_update_full_model=100,
            oracle=oracle,
            core_max=100,
            coverage_radius=40.0,
            loss_func=loss_func,
            smooth_hinge_theta=0.5,
            smooth_hinge_tau=0.5,
            random_state=3333,
            **candidate_params,
        )
        print('Running ...')
        clf.fit(x_total, y_total)
        print('Mistake rate: {0:.2f}%, Training time: {1} seconds'.format(clf.mistake_rate * 100, int(clf.train_time)))
        mistake_rate_avg += clf.mistake_rate
        train_time_avg += clf.train_time
    return mistake_rate_avg / num_runs, train_time_avg / num_runs, candidate_params


mistake_rate_lst = []
run_param_lst = []
time_lst = []


def log_result(result):
    mistake_rate, time, params = result
    mistake_rate_lst.append(mistake_rate)
    time_lst.append(time)
    run_param_lst.append(params)

    print("\n========== FINAL RESULT ==========")
    print('Data set: {}'.format(dataset))
    print('Oracle: {}'.format(oracle))
    print('Loss function: {}'.format(loss_func))
    print('Mistake rate: {0:.2f}%\nTraining time: {1} seconds'.format(mistake_rate * 100, int(time)))


def run_grid_search_multicore():
    if not os.path.exists(os.path.join(data_dir(), dataset + '_train.libsvm')):
        dataset_info = data_info(dataset)
        get_file(dataset, origin=dataset_info['origin'], untar=True, md5_hash=dataset_info['md5_hash'])
    params = {'regular_param': [0.000247295208655332],
              'gamma': [1.0]}
    candidate_params_lst = list(ParameterGrid(params))

    pool = mp.Pool() # maximum of workers
    for candidate_params in candidate_params_lst:
        pool.apply_async(run_one_candidate, args=(candidate_params,), callback=log_result)

    pool.close()
    pool.join()

    if len(candidate_params_lst) > 1:
        print("========== FINAL RESULT ==========")
        idx_best = np.argmin(np.array(mistake_rate_lst))
        print('Data set: {}'.format(dataset))
        print('Oracle: {}'.format(oracle))
        print('Loss func: {}'.format(loss_func))
        print('Best mistake rate: {}'.format(mistake_rate_lst[idx_best]))
        print('Best params: {}'.format(run_param_lst[idx_best]))
        print('Time per candidate param: {}'.format(time_lst[idx_best]))


if __name__ == '__main__':
    # pytest.main([__file__])
    # test_sgd_visualization_2d(block_figure_on_end=True)
    # run_gridsearch_single_core()
    run_grid_search_multicore()

