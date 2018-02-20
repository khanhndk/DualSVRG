from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import sys
import socket
import scipy.sparse

from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import ParameterGrid

import numpy as np

from male.configs import random_seed
from male.configs import data_dir
from male.callbacks import Display

from male.utils.data_utils import get_file
from male.utils.data_utils import data_info

import multiprocessing as mp


def test_visualization_2d(
        create_obj_func,
        show=False, block_figure_on_end=False,
        freq_predict_display=5,
        show_loss_display=False,
        grid_size=100,
        marker_size=10):
    file_name = os.path.join(data_dir(), 'demo/synthetic_2D_data_train.libsvm')
    x_train, y_train = load_svmlight_file(file_name)
    x_train = x_train.toarray()

    print('num_samples: {}'.format(x_train.shape[0]))

    predict_display = Display(
        freq=freq_predict_display,
        dpi='auto',
        show=show,
        block_on_end=block_figure_on_end,
        monitor=[{'metrics': ['predict'],
                  'title': "Visualization",
                  'xlabel': "X1",
                  'ylabel': "X2",
                  'grid_size': grid_size,
                  'marker_size': marker_size,
                  'left': None,
                  'right': None,
                  'top': None,
                  'bottom': None
                  }]
    )

    loss_display = Display(
        freq=1,
        dpi=72,
        show=show,
        block_on_end=block_figure_on_end,
        monitor=[{'metrics': ['train_loss'],
                  'type': 'line',
                  'title': "Learning losses",
                  'xlabel': "data points",
                  'ylabel': "loss",
                  }]
    )

    callbacks = [predict_display]
    if show_loss_display:
        callbacks.append(loss_display)

    users_params = {
        'callbacks': callbacks,
        'metrics': ['train_loss'],
    }

    learner = create_obj_func(users_params)
    learner.fit(x_train, y_train)
    y_train_pred = learner.predict(x_train)
    print("Training error = %.4f" % (1 - metrics.accuracy_score(y_train, y_train_pred)))


def test_real_dataset(create_obj_func, data_name=None, show=False, block_figure_on_end=False):
    if data_name is None:
        if len(sys.argv) > 2:
            data_name = sys.argv[2]
        else:
            raise Exception('Not specify dataset')

    np.random.seed(1234)

    print("========== Test on real data ==========")

    train_file_name = os.path.join(data_dir(), data_name + '_train.libsvm')
    test_file_name = os.path.join(data_dir(), data_name + '_test.libsvm')

    print(train_file_name)
    print(test_file_name)

    if not os.path.exists(train_file_name):
        raise Exception('File not found')
    if not os.path.exists(test_file_name):
        raise Exception('File not found')

    x_train, y_train = load_svmlight_file(train_file_name)
    x_test, y_test = load_svmlight_file(test_file_name, n_features=x_train.shape[1])

    users_params = dict()
    users_params = parse_arguments(users_params)
    print('users_params:', users_params)
    if 'sparse' not in users_params.keys():
        x_train = x_train.toarray()
        x_test = x_test.toarray()
        x = np.vstack((x_train, x_test))

    else:
        x = scipy.sparse.vstack((x_train, x_test))

    y = np.concatenate((y_train, y_test))
    cv = [-1] * x_train.shape[0] + [1] * x_test.shape[0]

    loss_display = Display(
        freq=1,
        dpi=72,
        show=show,
        block_on_end=block_figure_on_end,
        monitor=[{'metrics': ['train_loss', 'valid_loss'],
                  'type': 'line',
                  'title': "Learning losses in " + data_name,
                  'xlabel': "data points",
                  'ylabel': "loss",
                  }]
    )

    users_params['callbacks'] = [loss_display]
    users_params['metrics'] = ['train_loss', 'valid_loss']
    users_params['cv'] = cv

    learner = create_obj_func(users_params)
    learner.fit(x, y)

    y_train_pred = learner.predict(x_train)
    print('y_train:', np.unique(y_train))
    print('y_train_pred:', np.unique(y_train_pred))
    y_test_pred = learner.predict(x_test)
    print('y_test:', np.unique(y_test))
    print('y_test_pred:', np.unique(y_test_pred))

    print("Training acc = %.4f" % (metrics.accuracy_score(y_train, y_train_pred)))
    print("Testing acc = %.4f" % (metrics.accuracy_score(y_test, y_test_pred)))


def run_one_candidate(
        create_obj_func, candidate_params, data_name, attribute_names, file_config, num_runs,
        cross, num_features, keep_vars, ind_test=None, grid_search=True, max_torrance=1, online=False):

    # print('OS:', os.name)
    if (os.name == "posix") and file_config is not None:
        print(file_config)
        file = open(file_config, 'r')
        cpu_config = file.read()
        file.close()
        os.system("taskset -p " + cpu_config + " %d" % os.getpid())

    np.random.seed(random_seed())

    train_acc_avg = 0
    test_acc_avg = 0
    train_time_avg = 0
    mistake_rate_avg = 0

    log_lst = []
    total_runs = num_runs
    if cross > 0:
        total_runs = num_runs * cross

    for ri in range(total_runs):
        print('----------------------------------')
        if cross > 0:
            if ri % num_runs == 0:
                np.random.seed(1010)
            crossid = str(int(ri / num_runs))
            print('Run #{0} - Cross #{1}:'.format(ri+1, crossid+1))
            train_file_name = os.path.join(data_dir(), data_name + '_' + crossid + '.train.txt')
            test_file_name = os.path.join(data_dir(), data_name + '_' + crossid + '.test.txt')
            if ind_test is not None:
                ind_test_file_name = os.path.join(data_dir(), ind_test + '_test.libsvm')

            print('Train file:', train_file_name)
            print('Valid file:', test_file_name)
            if ind_test is not None:
                print(ind_test_file_name)
        else:
            print('Run #{0}:'.format(ri+1))
            train_file_name = os.path.join(data_dir(), data_name + '_train.libsvm')
            test_file_name = os.path.join(data_dir(), data_name + '_test.libsvm')

        if not os.path.exists(train_file_name):
            print('File ' + train_file_name + 'not found')
            raise Exception('File ' + train_file_name + ' not found')
        if not os.path.exists(test_file_name):
            raise Exception('File ' + test_file_name + ' not found')
        if ind_test is not None:
            if not os.path.exists(ind_test_file_name):
                raise Exception('File ' + ind_test_file_name + 'not found')

        if num_features is None:
            x_train, y_train = load_svmlight_file(train_file_name)
            x_test, y_test = load_svmlight_file(test_file_name)
            if ind_test is not None:
                x_ind_test, y_ind_test = load_svmlight_file(ind_test_file_name)
        else:
            x_train, y_train = load_svmlight_file(train_file_name, n_features=num_features)
            x_test, y_test = load_svmlight_file(test_file_name, n_features=num_features)
            if ind_test is not None:
                x_ind_test, y_ind_test = load_svmlight_file(ind_test_file_name, n_features=num_features)

        if grid_search:
            print('Trial params:', dict2string(candidate_params))
        learner = create_obj_func(candidate_params)
        if not hasattr(learner, 'sparse') or not learner.sparse:
            x_train = x_train.toarray()
            x_test = x_test.toarray()
            if ind_test is not None:
                x_ind_test = x_ind_test.toarray()

        if online:
            x_total = np.vstack((x_train, x_test))
            y_total = np.concatenate((y_train, y_test))
            print('Num total samples: {}'.format(x_total.shape[0]))
            print('Running ...')
            learner.fit(x_total, y_total)
        else:
            print('Num samples: {}'.format(x_train.shape[0]))
            print('Training ...')
            learner.fit(x_train, y_train)

        if online:
            mistake_rate = learner.mistake_rate
            mistake_rate_avg += mistake_rate
        else:
            y_train_pred = learner.predict(x_train)
            y_test_pred = learner.predict(x_test)

            train_labels, train_ycount = np.unique(y_train, return_counts=True)
            train_acc = metrics.accuracy_score(y_train, y_train_pred)
            train_acc_detail = np.diagonal(metrics.confusion_matrix(y_train, y_train_pred, train_labels)) / train_ycount

            test_labels, test_ycount = np.unique(y_test, return_counts=True)
            test_acc_detail = np.diagonal(metrics.confusion_matrix(y_test, y_test_pred, test_labels)) / test_ycount
            test_acc = metrics.accuracy_score(y_test, y_test_pred)

            if ind_test is not None:
                y_ind_test_pred = learner.predict(x_ind_test)
                ind_test_labels, ind_test_ycount = np.unique(y_ind_test, return_counts=True)
                ind_test_acc_detail = \
                    np.diagonal(metrics.confusion_matrix(y_ind_test, y_ind_test_pred, ind_test_labels)) / ind_test_ycount
                ind_test_acc = metrics.accuracy_score(y_ind_test, y_ind_test_pred)

        train_time = learner.train_time
        if online:
                print('Mistake rate: {0:.2f}%, Training time: {1} seconds'.format(mistake_rate*100, int(train_time)))
        else:
            if grid_search:
                print('Err on valid set: {0:.2f}%, Err on training set: {1:.2f}%, Training time: {2} seconds'
                      .format(100 - test_acc * 100, 100 - train_acc * 100, int(train_time)))
            else:
                print('Err on testing set: {0:.2f}%, Err on training set: {1:.2f}%, Training time: {2} seconds'
                      .format(100 - test_acc * 100, 100 - train_acc * 100, int(train_time)))

            train_acc_avg += train_acc
            test_acc_avg += test_acc

        train_time_avg += train_time

        log_lst.append({k: learner.__dict__[k] for k in attribute_names})
        log_lst[len(log_lst) - 1]['dataset'] = data_name
        log_lst[len(log_lst) - 1]['train_time'] = train_time
        if online:
            log_lst[len(log_lst) - 1]['mistake_rate'] = mistake_rate
        else:
            log_lst[len(log_lst) - 1]['train_acc'] = train_acc
            log_lst[len(log_lst) - 1]['test_acc'] = test_acc
            log_lst[len(log_lst) - 1]['train_acc_detail'] = train_acc_detail
            log_lst[len(log_lst) - 1]['test_acc_detail'] = test_acc_detail
            if ind_test is not None:
                log_lst[len(log_lst) - 1]['independent_test_acc'] = ind_test_acc
                log_lst[len(log_lst) - 1]['independent_test_acc_detail'] = ind_test_acc_detail

        for key in keep_vars:
            candidate_params[key] = learner.__dict__[key]

        if not online:
            if (1-test_acc) > max_torrance:
                total_runs = 1
                break

    if online:
        return mistake_rate_avg / total_runs, train_time_avg / total_runs, log_lst, grid_search, candidate_params
    else:
        return \
            train_acc_avg / total_runs, test_acc_avg / total_runs, train_time_avg / total_runs, log_lst, \
            grid_search, candidate_params


mistake_rate_lst = []
train_acc_lst = []
test_acc_lst = []
run_param_lst = []
time_lst = []
testid_lst = []
online = False


def log_result(result):
    if online:
        mistake_rate, time, log_lst, grid_search, params = result
        mistake_rate_lst.append(mistake_rate)
    else:
        train_acc, test_acc, time, log_lst, grid_search, params = result
        train_acc_lst.append(train_acc)
        test_acc_lst.append(test_acc)
    time_lst.append(time)
    run_param_lst.append(params)
    testid_lst.append(len(testid_lst) + 1)

    if online:
        if grid_search:
            print('Mistake rate: {0:.2f}%, Training time: {1} seconds, Params: {2}'.
                  format(mistake_rate*100, int(time), dict2string(params)))
        else:
            print("\n========== FINAL RESULT ==========")
            print('Data set: {}'.format(log_lst[len(log_lst) - 1]['dataset']))
            print('Oracle: {}'.format(params['oracle']))
            print('Loss function: {}'.format(params['loss_func']))
            print('Mistake rate: {0:.2f}%\nTraining time: {1} seconds'.format(mistake_rate*100, int(time)))
    else:
        if grid_search:
            print('Err on valid set: {0:.2f}%, Err on training set: {1:.2f}%, Training time: {2} seconds, Params: {3}'.
                  format(100-test_acc*100, 100-train_acc*100, int(time), dict2string(params)))
        else:
            print("\n========== FINAL RESULT ==========")
            print('Data set: {}'.format(log_lst[len(log_lst) - 1]['dataset']))
            print('Oracle: {}'.format(params['oracle']))
            print('Loss function: {}'.format(params['loss_func']))
            print('Err on testing set: {0:.2f}%\nErr on training set: {1:.2f}%\nTraining time: {2} seconds'
                  .format(100-test_acc*100, 100-train_acc*100, int(time)))

    testid = len(testid_lst)
    log_filename = \
        log_lst[len(log_lst) - 1]['model_name'] + '.' + log_lst[len(log_lst) - 1]['dataset'] \
        + '.' + socket.gethostname() + '.txt'
    log_file = open(log_filename, 'a')
    for it in range(len(log_lst)):
        log_file.write('testid:' + str(testid) + '\trunid:' + str(it) + '\t')
        if online:
            log_file.write('mistake_rate:' + str(log_lst[it]['mistake_rate']) + '\t')
        else:
            log_file.write('test_acc:' + str(log_lst[it]['test_acc']) + '\t')
            log_file.write('train_acc:' + str(log_lst[it]['train_acc']) + '\t')
        log_file.write('time:' + str(log_lst[it]['train_time']) + '\t')
        for key, value in log_lst[it].items():
            if type(value) is np.ndarray:
                log_file.write(key + ':' + '|'.join('{0:.4f}'.format(x) for x in value.ravel()) + '\t')
            else:
                log_file.write(key + ':' + str(value) + '\t')
        log_file.write('\n')
    log_file.close()


def run_grid_search_multicore(
        create_obj_func, params_gridsearch, attribute_names, dataset=None, num_workers=4, file_config=None,
        num_runs=3, cross=0, num_features=None, full_dataset=None, keep_vars=[], ind_test=None, max_torrance=1):
    if dataset is None:
        if len(sys.argv) > 2:
            dataset = sys.argv[2]
        else:
            raise Exception('Not specify dataset')

    params_gridsearch = parse_arguments(params_gridsearch, True)
    # print(params_gridsearch)
    file_config, params_gridsearch = extract_param('file_config', file_config, params_gridsearch)
    num_workers, params_gridsearch = extract_param('num_workers', num_workers, params_gridsearch)
    num_runs, params_gridsearch = extract_param('num_runs', num_runs, params_gridsearch)
    cross, params_gridsearch = extract_param('cross', cross, params_gridsearch)
    num_features, params_gridsearch = extract_param('num_features', num_features, params_gridsearch)
    full_dataset, params_gridsearch = extract_param('full_dataset', full_dataset, params_gridsearch)
    ind_test, params_gridsearch = extract_param('ind_test', ind_test, params_gridsearch)
    max_torrance, params_gridsearch = extract_param('max_torrance', max_torrance, params_gridsearch)
    if ind_test is not None:
        if full_dataset is None:
            ind_test = dataset
        else:
            ind_test = full_dataset

    if full_dataset is None:
        full_dataset = dataset

    if not os.path.exists(os.path.join(data_dir(), full_dataset + '_train.libsvm')):
        dataset_info = data_info(full_dataset)
        get_file(full_dataset, origin=dataset_info['origin'], untar=True, md5_hash=dataset_info['md5_hash'])

    candidate_params_lst = list(ParameterGrid(params_gridsearch))
    grid_search = True
    if len(candidate_params_lst) == 1:
        grid_search = False

    pool = mp.Pool(num_workers)  # maximum of workers
    result_lst = []
    for candidate_params in candidate_params_lst:
        result = pool.apply_async(
            run_one_candidate,
            args=(
                create_obj_func, candidate_params, dataset, attribute_names, file_config, num_runs, cross,
                num_features, keep_vars, ind_test, grid_search, max_torrance, online),
            callback=log_result
        )
        result_lst.append(result)

    for result in result_lst:
        result.get()
    pool.close()
    pool.join()

    if len(candidate_params_lst) > 1:
        print("========== FINAL RESULT ==========")
        if online:
            idx_best = np.argmin(np.array(mistake_rate_lst))
        else:
            idx_best = np.argmax(np.array(test_acc_lst))
        print('Data set: {}'.format(dataset))
        print('Best testid: {}'.format(testid_lst[idx_best]))
        if online:
            print('Best mistake rate: {}'.format(mistake_rate_lst[idx_best]))
        else:
            print('Best err on training set: {}'.format(1-train_acc_lst[idx_best]))
            print('Best err on valid set: {}'.format(1-test_acc_lst[idx_best]))
        print('Best params: {}'.format(run_param_lst[idx_best]))

        if cross > 0:
            print('Run the best one')
            num_runs_for_best = num_runs
            if num_runs < 3:
                num_runs_for_best = 3
            best_result = run_one_candidate(
                create_obj_func, run_param_lst[idx_best], full_dataset, attribute_names, file_config, num_runs_for_best,
                cross=0, num_features=num_features, keep_vars=keep_vars, online=online)
            # best_result['gridsearch_time'] = np.sum(np.array(time_lst))
            log_result(best_result)


def main_func(
        create_obj_func,
        choice_default=0,
        dataset_default='svmguide1',
        params_gridsearch=None,
        attribute_names=None,
        num_workers=4,
        file_config=None,
        run_exp=False,
        keep_vars=[],
        run_online=False,
        **kwargs):
    global online
    online = run_online

    if not run_exp:
        choice_lst = [0, 1, 2]
        data_name = dataset_default
    elif len(sys.argv) > 1:
        choice_lst = [int(sys.argv[1])]
        data_name = None
    else:
        choice_lst = [choice_default]
        data_name = dataset_default

    for choice in choice_lst:
        if choice == 0:
            test_visualization_2d(create_obj_func, show=run_exp, block_figure_on_end=run_exp, **kwargs)
        elif choice == 1:
            test_real_dataset(create_obj_func, data_name, show=run_exp, block_figure_on_end=run_exp)
        else:
            run_grid_search_multicore(
                create_obj_func, params_gridsearch, attribute_names, data_name, num_workers, file_config,
                keep_vars=keep_vars
            )


def parse_arguments(params, as_array=False):
    for it in range(3, len(sys.argv), 2):
        params[sys.argv[it]] = parse_argument(sys.argv[it + 1], as_array)
    return params


def parse_argument(string, as_array=False):
    try:
        result = int(string)
    except ValueError:
        try:
            result = float(string)
        except ValueError:
            if str.lower(string) == 'true':
                result = True
            elif str.lower(string) == 'false':
                result = False
            elif ('|' in string) and ('[' in string) and (']' in string):
                result = [float(item) for item in string[1:-1].split('|')]
                return result
            elif (',' in string) and ('(' in string) and (')' in string):
                split = string[1:-1].split(',')
                result = float(split[0]) ** np.arange(float(split[1]), float(split[2]), float(split[3]))
                return result
            else:
                result = string

    return [result] if as_array else result


def resolve_conflict_params(primary_params, secondary_params):
    for key in primary_params.keys():
        if key in secondary_params.keys():
            del secondary_params[key]
    return secondary_params


def extract_param(key, value, params_gridsearch):
    if key in params_gridsearch.keys():
        value = params_gridsearch[key][0]
        del params_gridsearch[key]
    return value, params_gridsearch


def dict2string(params):
    result = ''
    for key, value in params.items():
        if type(value) is np.ndarray:
            if value.size < 16:
                result += key + ': ' + '|'.join('{0:.4f}'.format(x) for x in value.ravel()) + ', '
        else:
            result += key + ': ' + str(value) + ', '
    return '{' + result[:-2] + '}'
