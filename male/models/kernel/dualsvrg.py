from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from collections import deque

from male import Model
from male.utils.generic_utils import Progbar


class DualSVRG(Model):
    """ Stochastic Variance-reduced Gradient Descent for Kernel Online Learning
        Speedup Version
    """

    def __init__(self,
                 model_name="DualSVRG",
                 regular_param=0.1,
                 learning_rate_scale=0.8,
                 gamma=10,
                 rf_dim=400,
                 num_epochs=1,
                 batch_size=1,
                 cache_size=100,
                 freq_update_full_model=100,
                 oracle='budget',
                 core_max=-1,
                 coverage_radius=-1,
                 loss_func='hinge',
                 smooth_hinge_theta=0.5,
                 smooth_hinge_tau=0.5,
                 freq_calc_metrics=300,
                 show_loss=True,
                 info=0,
                 **kwargs):
        super(DualSVRG, self).__init__(model_name=model_name, **kwargs)
        self.regular_param = regular_param
        self.learning_rate_scale = learning_rate_scale
        self.gamma = gamma
        self.rf_dim = rf_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.freq_update_full_model = freq_update_full_model
        self.oracle = oracle
        self.core_max = core_max
        self.coverage_radius = coverage_radius
        self.loss_func = loss_func
        self.smooth_hinge_theta = smooth_hinge_theta
        self.smooth_hinge_tau = smooth_hinge_tau
        self.freq_calc_metrics = freq_calc_metrics
        self.show_loss = show_loss
        self.info = info

    def _init(self):
        super(DualSVRG, self)._init()

        learning_rate_m = 1 if self.loss_func == 'hinge' else 1.0 / self.smooth_hinge_tau
        self.learning_rate = self.learning_rate_scale * np.minimum(
            1.0 / learning_rate_m,
            1.0 / np.abs(12*learning_rate_m-self.regular_param))

        self.omega = None
        self.w_cur_core = None
        self.num_core = None
        self.idx_core = None
        self.x_core = None
        self.w_cur_rf = None

        self.w_full_core = None
        self.w_full_rf = None

        self.mistake_rate = 0
        self.rf_2dim = self.rf_dim * 2
        self.rf_2dim_pad = self.rf_2dim + 1
        self.rf_scale = 1.0 / np.sqrt(self.rf_dim+1)

    def _calc_rf(self, xn):
        omega_x = np.matmul(xn, self.omega)
        # omega_x = np.sum(self.omega * xn, axis=1)
        xn_rf = np.ones(self.rf_2dim_pad)
        xn_rf[0:self.rf_dim] = np.cos(omega_x)
        xn_rf[self.rf_dim: self.rf_2dim] = np.sin(omega_x)
        xn_rf *= self.rf_scale
        return xn_rf

    def _get_grad_full(self, xn, yn):
        yn_pred, wxn, xn_rf, wxn_rf, dist2_xn, kn, wxn_core = self._predict_one_given_w(
            self.w_full_rf, self.w_full_core, self.num_core, xn
        )
        return self._get_grad(wxn, yn)

    def _get_grad_full_pre_calc_rf(self, xn, yn, n, kn):
        xn_rf = self.x_rf[n, :]
        yn_pred, wxn, wxn_rf, wxn_core, kn = self._predict_one_given_w_pre_calc_rf_kn(
            xn, self.w_full_rf, self.w_full_core, self.num_core, xn_rf, kn
        )
        return self._get_grad(wxn, yn), kn

    def _get_grad(self, wxn, yn):
        idx_runner = np.argmax(wxn[np.arange(self.num_classes) != yn])
        idx_runner += (idx_runner >= yn)
        wxn_runner = wxn[idx_runner]
        wxn_true = wxn[yn]
        o = wxn_true - wxn_runner

        if self.loss_func == 'hinge':
            if o < 1-self.smooth_hinge_tau:
                loss = 1 - o - 0.5 * self.smooth_hinge_tau
                grad = -1
            elif o <= 1:
                loss = (0.5 / self.smooth_hinge_tau) * ((1 - o)**2)
                grad = 1.0 / self.smooth_hinge_tau
            else:
                loss = 0
                grad = 0
        elif self.loss_func == 'logistic':
            if o < -500:
                grad = 1
                loss = -o
            elif o > 500:
                grad = 0
                loss = 0
            else:
                exp_minus_o = np.exp(-o)
                grad = - exp_minus_o / (exp_minus_o + 1)
                loss = np.log(1+exp_minus_o)
        else:
            raise NotImplementedError
        return grad, idx_runner, loss

    def _get_dist2(self, xn, start_idx=0):
        dist2 = np.sum(
            (self.x_[self.idx_core[start_idx:self.num_core], :] - xn) ** 2, axis=1)
        return dist2

    def _oracle_always(self, dist2_xn, n):
        return True

    def _oracle_budget(self, dist2_xn, n):
        if n > self.core_max:
            return True
        else:
            return False

    def _oracle_coverage(self, dist2_xn, n):
        if np.any(dist2_xn < self.coverage_radius):
            return True
        else:
            return False

    def _predict_one(self, xn):
        return self._predict_one_given_w(self.w_cur_rf, self.w_cur_core, self.num_core, xn)

    def _predict_one_pre_calc_rf(self, xn, n):
        xn_rf = self.x_rf[n, :]
        return self._predict_one_given_w_pre_calc_rf(self.w_cur_rf, self.w_cur_core, self.num_core, xn, xn_rf)

    def _predict_one_given_w(self, w_rf, w_core, num_core, xn):
        xn_rf = self._calc_rf(xn)
        return self._predict_one_given_w_pre_calc_rf(w_rf, w_core, num_core, xn, xn_rf)

    def _predict_one_given_w_pre_calc_rf(self, w_rf, w_core, num_core, xn, xn_rf):
        wxn_rf = np.sum(w_rf * xn_rf, axis=1)

        dist2_xn = self._get_dist2(xn)
        kn = np.exp(-self.gamma * dist2_xn)
        wxn_core = np.sum(w_core[:, :num_core] * kn, axis=1)

        wxn = wxn_rf + wxn_core
        y_pred = np.argmax(wxn)
        return y_pred, wxn, xn_rf, wxn_rf, dist2_xn, kn, wxn_core

    def _predict_one_given_w_pre_calc_rf_kn(self, xn, w_rf, w_core, num_core, xn_rf, kn):
        wxn_rf = np.sum(w_rf * xn_rf, axis=1)

        if len(kn) < num_core:
            dist2_app = self._get_dist2(xn, len(kn))
            kn_app = np.exp(-self.gamma * dist2_app)
            kn = np.append(kn, kn_app)

        wxn_core = np.sum(w_core[:, :num_core] * kn, axis=1)

        wxn = wxn_rf + wxn_core
        y_pred = np.argmax(wxn)
        return y_pred, wxn, wxn_rf, wxn_core, kn

    def _get_mean_loss(self, x, y):
        num_tests = x.shape[0]
        mean_loss = 0.0
        for nn in range(num_tests):
            xn = x[nn, :]
            yn = y[nn]
            yn_pred, wxn, xn_rf, wxn_rf, dist2_xn, kn, wxn_core = self._predict_one(xn)
            grad_cur, idx_cur_runner, loss_cur = self._get_grad(wxn, yn)
            mean_loss += np.maximum(0, loss_cur)
        mean_loss = mean_loss / num_tests
        return mean_loss

    def _get_dual_wnorm2(self):
        dist2 = np.zeros((self.num_core, self.num_core))
        for co in range(self.num_core):
            dist2[co, :] = self._get_dist2(self.x_[self.idx_core[co], :])
        k = np.exp(-self.gamma * dist2)
        dual_wnorm2 = 0
        for ci in range(self.num_classes):
            w_core_norm2_class = np.kron(
                self.w_cur_core[ci, :self.num_core], self.w_cur_core[ci, :self.num_core]).reshape(
                (self.num_core, self.num_core)
            )
            w_core_norm2_class *= k
            w_rf_norm2_class = np.sum(self.w_cur_rf[ci, :]**2)
            dual_wnorm2 += np.sum(w_core_norm2_class) + w_rf_norm2_class

        return dual_wnorm2

    def _fit_loop_v1(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        if self.oracle == 'budget':
            self._fit_loop_budget(x, y, do_validation, x_valid, y_valid, callbacks, callback_metrics)
        else:
            self._fit_loop_cov_alw(x, y, do_validation, x_valid, y_valid, callbacks, callback_metrics)

    def _fit_loop_budget(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        self.x_ = x
        self.y_ = y

        num_samples = x.shape[0]
        input_dim = x.shape[1]
        rf_dim = self.rf_dim
        rf_2dim_pad = self.rf_2dim_pad

        self.omega = np.random.normal(0, self.gamma / 2, (input_dim, rf_dim))
        self.w_cur_core = np.zeros((self.num_classes, self.core_max+1))
        self.idx_core = np.zeros(self.core_max+1, dtype=int)
        chk_core = np.zeros(num_samples+1, dtype=int)
        self.num_core = 0
        self.w_cur_rf = np.zeros((self.num_classes, rf_2dim_pad))
        self.mistake_rate = 0

        self.w_full_core = np.zeros((self.num_classes, num_samples+1))
        self.w_full_rf = np.zeros((self.num_classes, rf_2dim_pad))

        self.w_cur_core[0] = 0
        self.w_full_core[0] = 0
        self.idx_core[0] = 0
        self.num_core += 1

        cum_grad_rf = np.zeros((self.num_classes, self.rf_2dim_pad))
        idx_grad_core_lst = []  # np.zeros(self.batch_size*2 + 1)
        grad_core_lst = []  # np.zeros((self.num_classes, self.batch_size * 2 + 1))
        num_grad_core_lst = 0
        w_sum_rf = np.zeros((self.num_classes, self.rf_2dim_pad))
        w_sum_core = np.zeros((self.num_classes, num_samples+1))
        i_batch = 0
        xnt_rf_lst = np.zeros((self.cache_size, self.rf_2dim_pad))

        max_loop = self.num_epochs * num_samples
        idx_samples = np.zeros(max_loop, dtype=int)
        idx_remove = np.zeros(max_loop, dtype=int)
        for n in range(max_loop):
            callbacks.on_epoch_begin(self.epoch)
            nt = np.random.randint(num_samples)
            xnt = x[nt, :]
            ynt = y[nt]
            idx_samples[n] = nt

            dist_xnt = self._get_dist2(xnt)

            if self.num_core > self.core_max:
                idx_core_remove = np.argmin(np.sum(np.abs(self.w_cur_core[:, :self.num_core]), axis=0))
                nt_remove = self.idx_core[idx_core_remove]
                idx_remove[n] = nt_remove
                self.w_cur_core[:, idx_core_remove] = 0

                try:
                    idx_grad_remove = idx_grad_core_lst.index(nt_remove)
                    del idx_grad_core_lst[idx_grad_remove]
                    del grad_core_lst[idx_grad_remove]
                    num_grad_core_lst -= 1
                except ValueError:
                    tmp = 0
                    # print('Warning: not found in idx_grad_core')

                vnt_core_cur = self._get_grad_core(
                    self.w_cur_core, self.num_core, dist_xnt, ynt)
                vnt_core_full = self._get_grad_core(
                    self.w_full_core, self.num_core, dist_xnt, ynt)

                # self.w_cur_core -= \
                #     self.regular_param * self.w_full_core \
                #     / (self.learning_rate * self.regular_param + 1)

                chk_core[self.idx_core[idx_core_remove]] = 0
                self.idx_core[idx_core_remove] = nt
                chk_core[nt] = idx_core_remove

                self.w_cur_core[:, idx_core_remove] = \
                    - (vnt_core_cur - vnt_core_full) * self.learning_rate \
                    / (self.learning_rate * self.regular_param + 1)
                # CARE when upgrade to BATCH SETTING += NOT = : NO NEED in this case

                num_grad_core_lst += 1
                idx_grad_core_lst.append(nt)
                grad_core_lst.append(vnt_core_full)
                self.w_cur_core[:, self.num_core - num_grad_core_lst: self.num_core] += \
                    - np.array(grad_core_lst).T * self.learning_rate\
                    / (num_grad_core_lst * (self.learning_rate * self.regular_param + 1))

                w_sum_core[:, :self.num_core] += self.w_cur_core[:, :self.num_core]

                xnt_rf = self._calc_rf(x[nt_remove])
                xnt_rf_lst[i_batch, :] = xnt_rf

                vnt_rf_cur = self._get_grad_rf(self.w_cur_rf, xnt_rf, y[nt_remove])
                vnt_rf_full = self._get_grad_rf(self.w_full_rf, xnt_rf, y[nt_remove])
                cum_grad_rf += vnt_rf_full
                vnt = \
                    vnt_rf_cur - vnt_rf_full \
                    + cum_grad_rf / self.cache_size
                self.w_cur_rf = \
                    (self.w_cur_rf - self.learning_rate * vnt) / \
                    (self.learning_rate * self.regular_param + 1)
                w_sum_rf += self.w_cur_rf

            else:
                # print('add core')
                # add to core
                vnt_core_cur = self._get_grad_core(
                    self.w_cur_core, self.num_core, dist_xnt, ynt)
                vnt_core_full = self._get_grad_core(
                    self.w_full_core, self.num_core, dist_xnt, ynt)

                # self.w_cur_core -= \
                #     self.regular_param * self.w_full_core \
                #     / (self.learning_rate * self.regular_param + 1)

                if chk_core[nt] == 0:
                    self.num_core += 1
                    self.idx_core[self.num_core - 1] = nt
                    chk_core[nt] = self.num_core - 1
                    self.w_cur_core[:, self.num_core - 1] = np.zeros(self.num_classes)

                self.w_cur_core[:, self.num_core - 1] += \
                    - (vnt_core_cur - vnt_core_full) * self.learning_rate \
                    / (self.learning_rate * self.regular_param + 1)
                # CARE when upgrade to BATCH SETTING += NOT =

                num_grad_core_lst += 1
                grad_core_lst.append(vnt_core_full)
                # grad_core_lst[:, num_grad_core_lst-1] = vnt_core_full
                idx_grad_core_lst.append(nt)
                if self.num_core >= num_grad_core_lst:
                    self.w_cur_core[:, self.num_core - num_grad_core_lst: self.num_core] += \
                        - np.array(grad_core_lst).T * self.learning_rate\
                        / (num_grad_core_lst * (self.learning_rate * self.regular_param + 1))
                else:
                    self.w_cur_core[:, :self.num_core] += \
                        - np.array(grad_core_lst)[:self.num_core, :].T * self.learning_rate \
                        / (num_grad_core_lst * (self.learning_rate * self.regular_param + 1))

                w_sum_core[:, :self.num_core] += self.w_cur_core[:, :self.num_core]

            # print('num_core=', self.num_core)
            if (n+1) % self.freq_update_full_model == 0:
                self.w_full_rf = w_sum_rf / self.freq_update_full_model
                self.w_cur_rf = self.w_full_rf.copy()
                w_sum_rf = np.zeros((self.num_classes, self.rf_2dim_pad))

                self.w_full_core = w_sum_core / self.freq_update_full_model
                self.w_cur_core = self.w_full_core.copy()
                w_sum_core = np.zeros((self.num_classes, num_samples+1))

                cum_grad_rf = np.zeros((self.num_classes, self.rf_2dim_pad))
                grad_core_lst = []
                idx_grad_core_lst = []
                num_grad_core_lst = 0
                for i in range(n-self.cache_size+1, n+1):
                    if i > self.core_max:
                        cum_grad_rf += self._get_grad_rf(
                            self.w_full_rf,
                            xnt_rf_lst[((i - n - 1 + self.cache_size) + i_batch + 1) % self.cache_size],
                            y[idx_remove[i]])

                    dist_xi = self._get_dist2(x[self.idx_core[i+self.num_core-n-1], :])
                    idx_grad_core_lst.append(self.idx_core[i+self.num_core-n-1])
                    grad_core_lst.append(self._get_grad_core(
                        self.w_full_core, self.num_core, dist_xi,
                        y[self.idx_core[i+self.num_core-n-1]]))
                    num_grad_core_lst += 1

            self.epoch += 1
            i_batch = (i_batch + 1) % self.cache_size
            callbacks.on_epoch_end(self.epoch)

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        self.x_ = x
        self.y_ = y

        num_samples = x.shape[0]
        input_dim = x.shape[1]
        rf_dim = self.rf_dim
        rf_2dim_pad = self.rf_2dim_pad

        self.omega = np.random.normal(0, self.gamma / 2, (input_dim, rf_dim))
        omega_x = np.matmul(x, self.omega)  # (num_samples, rf_dim)
        self.x_rf = np.hstack((np.cos(omega_x), np.sin(omega_x), np.ones((num_samples, 1))))
        # self.omega = np.random.normal(0, self.gamma / 2.0, (rf_dim, input_dim))
        self.w_cur_core = np.zeros((self.num_classes, num_samples))
        self.idx_core = np.zeros(num_samples, dtype=int)
        self.chk_core = -np.ones(num_samples, dtype=int)
        self.num_core = 0

        self.w_cur_rf = np.zeros((self.num_classes, rf_2dim_pad))

        self.w_full_core = np.zeros((self.num_classes, num_samples))
        self.w_full_rf = np.zeros((self.num_classes, rf_2dim_pad))

        self.w_cur_core[0] = 0
        self.w_full_core[0] = 0
        self.idx_core[0] = 0
        self.chk_core[0] = 0
        self.num_core += 1

        grad_lst = deque()
        xnt_rf_lst = deque()
        grad_idx_lst = deque()
        k_lst = deque()

        sum_grad_rf = np.zeros((self.num_classes, self.rf_2dim_pad))
        sum_grad_core = np.zeros((self.num_classes, num_samples))
        w_sum_rf = np.zeros((self.num_classes, self.rf_2dim_pad))
        w_sum_core = np.zeros((self.num_classes, num_samples))

        max_loop = self.num_epochs * num_samples
        move_decision = np.zeros(max_loop)
        idx_samples = np.zeros(max_loop, dtype=int)
        progbar = Progbar(max_loop, show_steps=1)
        for n in range(max_loop):
            progbar.update(n)
            epoch_logs = {}
            callbacks.on_epoch_begin(self.epoch)
            if (n % self.freq_calc_metrics) == 0:
                if 'train_loss' in callback_metrics:
                    mean_loss = self._get_mean_loss(x, y)
                    epoch_logs['train_loss'] = mean_loss
                if 'valid_loss' in callback_metrics:
                    y_valid_trans = self._transform_labels(y_valid)
                    mean_loss = self._get_mean_loss(x_valid, y_valid_trans)
                    epoch_logs['valid_loss'] = mean_loss
                if 'obj_func' in callback_metrics:
                    mean_loss = self._get_mean_loss(x, y)
                    dual_wnorm2 = self._get_dual_wnorm2()
                    obj_func = self.regular_param * dual_wnorm2 + mean_loss
                    epoch_logs['obj_func'] = obj_func

            nt = np.random.randint(num_samples)
            xnt = x[nt, :]
            ynt = y[nt]
            idx_samples[n] = nt

            # predict
            ynt_pred, wxnt, xnt_rf, wxnt_rf, dist2_xnt, knt, wxnt_core = self._predict_one_pre_calc_rf(xnt, nt)
            grad_cur, idx_cur_runner, loss_cur = self._get_grad(wxnt, ynt)
            (grad_full, idx_full_runner, loss_full), knt = self._get_grad_full_pre_calc_rf(xnt, ynt, nt, knt)

            move_decision[n] = {
                'budget': self._oracle_budget,
                'coverage': self._oracle_coverage,
                'always_move': self._oracle_always,
            }[self.oracle](dist2_xnt, n)

            if len(grad_lst) > self.cache_size - 1:
                xnt_rf_lst.popleft()
                idx_pop = grad_idx_lst.popleft()
                k_lst.popleft()
                grad_tmp = grad_lst.popleft()

                if len(grad_tmp.shape) == 2:
                    sum_grad_rf -= grad_tmp
                else:
                    sum_grad_core[:, self.chk_core[idx_pop]] -= grad_tmp

            xnt_rf_lst.append(xnt)
            grad_idx_lst.append(nt)
            k_lst.append(knt)

            if move_decision[n]:
                # approximate
                vnt_rf_cur = np.zeros((self.num_classes, self.rf_2dim_pad))
                vnt_rf_cur[ynt, :] = grad_cur * xnt_rf
                vnt_rf_cur[idx_cur_runner, :] = -grad_cur * xnt_rf

                vnt_rf_full = np.zeros((self.num_classes, self.rf_2dim_pad))
                vnt_rf_full[ynt, :] = grad_full * xnt_rf
                vnt_rf_full[idx_full_runner, :] = -grad_full * xnt_rf

                sum_grad_rf += vnt_rf_full
                grad_lst.append(vnt_rf_full)

                vnt = vnt_rf_cur - vnt_rf_full + sum_grad_rf / len(grad_lst)
                self.w_cur_rf = \
                    (self.w_cur_rf - self.learning_rate * vnt) / \
                    (self.learning_rate * self.regular_param + 1)

                self.w_cur_core[:, :self.num_core] += \
                    - sum_grad_core[:, :self.num_core] * self.learning_rate \
                    / (len(grad_lst) * (self.learning_rate * self.regular_param + 1))
            else:
                # add to core
                vnt_core_cur = np.zeros(self.num_classes)
                vnt_core_cur[ynt] = grad_cur
                vnt_core_cur[idx_cur_runner] = -grad_cur

                vnt_core_full = np.zeros(self.num_classes)
                vnt_core_full[ynt] = grad_full
                vnt_core_full[idx_full_runner] = -grad_full

                if self.chk_core[nt] < 0:
                    self.num_core += 1
                    self.idx_core[self.num_core - 1] = nt
                    self.chk_core[nt] = self.num_core - 1
                    self.w_cur_core[:, self.num_core - 1] = np.zeros(self.num_classes)

                sum_grad_core[:, self.chk_core[nt]] += vnt_core_full
                grad_lst.append(vnt_core_full)

                self.w_cur_core[:, self.chk_core[nt]] += \
                    - (vnt_core_cur - vnt_core_full) * self.learning_rate \
                    / (self.learning_rate * self.regular_param + 1)
                # CARE when upgrade to BATCH SETTING += NOT =

                self.w_cur_core[:, :self.num_core] += \
                    - sum_grad_core[:, :self.num_core] * self.learning_rate \
                    / (len(grad_lst) * (self.learning_rate * self.regular_param + 1))

                self.w_cur_rf += -sum_grad_rf * self.learning_rate \
                    / (len(grad_lst) * (self.learning_rate * self.regular_param + 1))

            w_sum_rf += self.w_cur_rf
            w_sum_core[:, :self.num_core] += self.w_cur_core[:, :self.num_core]
            # print('num_core=', self.num_core)

            if (n+1) % self.freq_update_full_model == 0:
                self.w_full_rf = w_sum_rf / self.freq_update_full_model
                # self.w_cur_rf = self.w_full_rf.copy()
                w_sum_rf = np.zeros((self.num_classes, self.rf_2dim_pad))

                self.w_full_core = w_sum_core / self.freq_update_full_model
                # self.w_cur_core = self.w_full_core.copy()
                w_sum_core = np.zeros((self.num_classes, num_samples))

                sum_grad_rf = np.zeros((self.num_classes, self.rf_2dim_pad))
                sum_grad_core = np.zeros((self.num_classes, num_samples))
                grad_lst.clear()

                for i in range(n-self.cache_size+1, n+1):
                    if n < self.cache_size - 1:
                        continue
                    it = idx_samples[i]
                    if it != grad_idx_lst[i-n+self.cache_size-1]:
                        print('Error idx')
                        raise Exception
                    xit_tmp = x[it, :]
                    yit_tmp = y[it]
                    kit_tmp = k_lst.popleft()
                    (grad_full, idx_full_runner, loss_full), kit_tmp = self._get_grad_full_pre_calc_rf(
                        xit_tmp, yit_tmp, it, kit_tmp)
                    # dist2_tmp = self._get_dist2(xit_tmp)
                    # kit_test = np.exp(-self.gamma * dist2_tmp)
                    # if np.abs(np.sum(kit_test - kit_tmp)) > 1e-3:
                    #     print(kit_test)
                    #     print(kit_tmp)
                    k_lst.append(kit_tmp)
                    # if move_decision[i]:
                    if self.chk_core[it] < 0:
                        vit_rf_full = np.zeros((self.num_classes, self.rf_2dim_pad))
                        vit_rf_full[ynt, :] = grad_full * self.x_rf[it, :]
                        vit_rf_full[idx_full_runner, :] = -grad_full * self.x_rf[it, :]

                        sum_grad_rf += vit_rf_full
                        grad_lst.append(vit_rf_full)
                    else:
                        vit_core_full = np.zeros(self.num_classes)
                        vit_core_full[yit_tmp] = grad_full
                        vit_core_full[idx_full_runner] = -grad_full

                        # if self.chk_core[nt] < 0:
                        #     self.num_core += 1
                        #     self.idx_core[self.num_core - 1] = nt
                        #     self.chk_core[nt] = self.num_core - 1
                        #     self.w_cur_core[:, self.num_core - 1] = np.zeros(self.num_classes)

                        sum_grad_core[:, self.chk_core[it]] += vit_core_full
                        grad_lst.append(vit_core_full)

            if (n % self.freq_calc_metrics) == 0:
                self.epoch += 1
                callbacks.on_epoch_end(self.epoch, epoch_logs)
        if self.info:
            print('\nnum_core=', self.num_core)
        print('\nTesting ...')
        self.w_cur_core = self.w_full_core.copy()
        self.w_cur_rf = self.w_cur_rf.copy()

    def predict(self, x):
        y = np.zeros(x.shape[0], dtype=int)
        for n in range(x.shape[0]):
            y[n], _, _, _, _, _, _ = self._predict_one(x[n])
            y[n] = self._decode_labels(y[n])
        return y

    def display_prediction(self, **kwargs):
        visualize_classification_prediction(self, self.x_, self.y_, **kwargs)

    def display(self, param, **kwargs):
        if param == 'predict':
            self.display_prediction(**kwargs)
        else:
            raise NotImplementedError

    def get_params(self, deep=True):
        out = super(DualSVRG, self).get_params(deep=deep)
        param_names = DualSVRG._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out
