from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import numpy as np

np.seterr(all='raise')

from scipy.misc import logsumexp
from scipy.optimize import check_grad

from .rrf import RRF
from ...utils.generic_utils import make_batches


class KMM(RRF):
    """Kernel Mixture Models

    """

    def __init__(self,
                 model_name="KMM",
                 gamma=(0.5, 1.0, 2, 4),
                 num_kernels=4,
                 temperature=1.0,
                 anneal_rate=0.95,
                 min_temperature=0.5,
                 momentum=0.0,
                 adam_update=True,
                 sampling_gumbel=True,
                 learning_rate_mu=0.0,
                 learning_rate_alpha=0.1,
                 alternative_update=False,
                 num_nested_epochs=5,
                 **kwargs):
        super(KMM, self).__init__(model_name=model_name, **kwargs)
        self.gamma = gamma
        self.num_kernels = num_kernels
        self.temperature = temperature
        self.anneal_rate = anneal_rate
        self.min_temperature = min_temperature
        self.adam_update = adam_update
        self.momentum = momentum
        self.sampling_gumbel = sampling_gumbel
        self.learning_rate_mu = learning_rate_mu
        self.num_nested_epochs = num_nested_epochs
        self.alternative_update = alternative_update
        self.learning_rate_alpha = learning_rate_alpha

    def _init(self):
        super(KMM, self)._init()
        self.gamma = self.gamma if isinstance(self.gamma, tuple) else (self.gamma,)
        self.mu = None
        self.alpha = None
        self.gumbel = None

    def _init_params(self, x):
        # temporarily place \gamma for initialization of FOGD and RRF
        gamma0 = self.gamma
        self.gamma = 1.0
        super(KMM, self)._init_params(x)
        # restore the \gamma for KMM
        self.gamma = gamma0
        self.alpha = np.log(1 / self.num_kernels) * np.ones(self.num_kernels)  # (M,)
        self.gumbel = self.random_engine.gumbel(0.0, 1.0, size=(self.D, self.num_kernels))  # (D,M)
        self.mu = np.zeros([self.num_kernels, self.num_features])  # (M,d)
        self.z = self._get_z()
        # self.gamma -> \sigma initial value
        self.gamma_ = (np.log(self.gamma)[:, np.newaxis]
                       * np.ones([self.num_kernels, self.num_features]))  # (M,d)

    def _get_pi(self, **kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha
        return np.exp(alpha - logsumexp(alpha))  # include normalization

    def _update_z(self):
        if self.sampling_gumbel:
            self.gumbel = self.random_engine.gumbel(0.0, 1.0, size=(self.D, self.num_kernels))
        self.z = self._get_z()

    def _get_z(self, **kwargs):
        return self._get_z_approx(**kwargs)

    def _get_z_approx(self, **kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha
        x = (alpha[np.newaxis, :] + self.gumbel) / self.temperature  # (D,M)
        return np.exp(x - logsumexp(x, axis=1, keepdims=True))  # (D,M)

    def _sample_z(self, **kwargs):
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha
        gumbel = kwargs['gumbel'] if 'gumbel' in kwargs else self.gumbel
        return np.argmax(gumbel + alpha[np.newaxis, :], axis=1)  # (D,)

    def get_grad(self, x, y, **kwargs):
        n = x.shape[0]  # batch size
        z = kwargs['z'] if 'z' in kwargs else self.z  # (D,)
        w = kwargs['w'] if 'w' in kwargs else self.w  # (2D,C)
        mu = kwargs['mu'] if 'mu' in kwargs else self.mu  # (M,d)
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_  # (M,d)

        phi = kwargs['phi'] if 'phi' in kwargs else self._get_phi(x, **kwargs)  # (N,2D)
        wx = kwargs['wx'] if 'wx' in kwargs else phi.dot(w)  # (N,C)

        dw = self.lbd * w  # (2D,C)

        if self.num_classes > 2:
            wxy, t = self._get_wxy(wx, y)
            if self.loss == 'hinge':
                d = (wxy[:, np.newaxis] < 1) * phi  # (N,2D)
                dphi = -w[:, y[wxy < 1]].T + w[:, t[wxy < 1]].T  # (N,2D)
                domega = self._get_domega(x[wxy < 1], phi[wxy < 1])  # (N,2D,d) (gradient of \omega)
            else:  # logit loss
                c = np.exp(-wxy - np.logaddexp(0, -wxy))[:, np.newaxis]
                d = c * phi
                dphi = -c * (w[:, y].T - w[:, t].T)  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            for i in range(self.num_classes):
                dw[:, i] += -d[y == i].sum(axis=0) / n
                dw[:, i] += d[t == i].sum(axis=0) / n
            dmu = self._get_dmu(domega, dphi, z) / n  # (M,d)
            dgamma = self._get_dgamma(gamma, domega, dphi, z) / n  # (M,d)
        else:
            if self.loss == 'hinge':
                wxy = y * wx
                dw += np.sum(-y[wxy < 1, np.newaxis] * phi[wxy < 1], axis=0) / n

                # compute gradients of \mu
                dphi = -y[wxy < 1, np.newaxis] * w  # (N,2D) (gradient of \Phi(x))
                domega = self._get_domega(x[wxy < 1], phi[wxy < 1])  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'l1':
                wxy = np.sign(wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'l2':
                wxy = (wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'logit':
                wxy = y * wx
                c = (-y * np.exp(-wxy - np.logaddexp(0, -wxy)))[:, np.newaxis]
                dw += np.mean(c * phi, axis=0)
                dphi = c * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'eps_insensitive':
                wxy = np.abs(y - wx) > self.eps
                c = np.sign(wx - y)[:, np.newaxis]
                d = c * phi
                dw += d[wxy].sum(axis=0) / n
                dphi = c[wxy] * w  # (N,2D)
                domega = self._get_domega(x[wxy], phi[wxy])  # (N,2D,d) (gradient of \omega)

            dmu = self._get_dmu(domega, dphi, z) / n  # (M,d)
            dgamma = self._get_dgamma(gamma, domega, dphi, z) / n  # (M,d)

        return dw, dmu, dgamma

    def get_grad_all(self, x, y, **kwargs):
        n = x.shape[0]  # batch size
        z = kwargs['z'] if 'z' in kwargs else self.z  # (D,)
        w = kwargs['w'] if 'w' in kwargs else self.w  # (2D,C)
        mu = kwargs['mu'] if 'mu' in kwargs else self.mu  # (M,d)
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_  # (M,d)
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha  # (M,)

        phi = kwargs['phi'] if 'phi' in kwargs else self._get_phi(x, **kwargs)  # (N,2D)
        wx = kwargs['wx'] if 'wx' in kwargs else phi.dot(w)  # (N,C)

        dw = self.lbd * w  # (2D,C)

        if self.num_classes > 2:
            wxy, t = self._get_wxy(wx, y)
            if self.loss == 'hinge':
                d = (wxy[:, np.newaxis] < 1) * phi  # (N,2D)
                dphi = -w[:, y[wxy < 1]].T + w[:, t[wxy < 1]].T  # (N,2D)
                domega = self._get_domega(x[wxy < 1], phi[wxy < 1])  # (N,2D,d) (gradient of \omega)
            else:  # logit loss
                c = np.exp(-wxy - np.logaddexp(0, -wxy))[:, np.newaxis]
                d = c * phi
                dphi = -c * (w[:, y].T - w[:, t].T)  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            for i in range(self.num_classes):
                dw[:, i] += -d[y == i].sum(axis=0) / n
                dw[:, i] += d[t == i].sum(axis=0) / n
            dmu = self._get_dmu(domega, dphi, z) / n  # (M,d)
            dgamma = self._get_dgamma(gamma, domega, dphi, z) / n  # (M,d)
            dalpha = self._get_dalpha(domega, dphi, z, mu, gamma) / n  # (M,)
        else:
            if self.loss == 'hinge':
                wxy = y * wx
                dw += np.sum(-y[wxy < 1, np.newaxis] * phi[wxy < 1], axis=0) / n

                # compute gradients of \mu
                dphi = -y[wxy < 1, np.newaxis] * w  # (N,2D) (gradient of \Phi(x))
                domega = self._get_domega(x[wxy < 1], phi[wxy < 1])  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'l1':
                wxy = np.sign(wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'l2':
                wxy = (wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'logit':
                wxy = y * wx
                c = (-y * np.exp(-wxy - np.logaddexp(0, -wxy)))[:, np.newaxis]
                dw += np.mean(c * phi, axis=0)
                dphi = c * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'eps_insensitive':
                wxy = np.abs(y - wx) > self.eps
                c = np.sign(wx - y)[:, np.newaxis]
                d = c * phi
                dw += d[wxy].sum(axis=0) / n
                dphi = c * w  # (N,2D)
                domega = self._get_domega(x[wxy], phi[wxy])  # (N,2D,d) (gradient of \omega)

            dmu = self._get_dmu(domega, dphi, z) / n  # (M,d)
            dgamma = self._get_dgamma(gamma, domega, dphi, z) / n  # (M,d)
            dalpha = self._get_dalpha(domega, dphi, z, mu, gamma) / n  # (M,)

        return dw, dmu, dgamma, dalpha

    def get_grad_mu_gamma_alpha(self, x, y, **kwargs):
        n = x.shape[0]  # batch size
        z = kwargs['z'] if 'z' in kwargs else self.z  # (D,)
        w = kwargs['w'] if 'w' in kwargs else self.w  # (2D,C)
        mu = kwargs['mu'] if 'mu' in kwargs else self.mu  # (M,d)
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_  # (M,d)
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha  # (M,)

        phi = kwargs['phi'] if 'phi' in kwargs else self._get_phi(x, **kwargs)  # (N,2D)
        wx = kwargs['wx'] if 'wx' in kwargs else phi.dot(w)  # (N,C)

        if self.num_classes > 2:
            wxy, t = self._get_wxy(wx, y)
            if self.loss == 'hinge':
                dphi = -w[:, y[wxy < 1]].T + w[:, t[wxy < 1]].T  # (N,2D)
                domega = self._get_domega(x[wxy < 1], phi[wxy < 1])  # (N,2D,d) (gradient of \omega)
            else:  # logit loss
                c = np.exp(-wxy - np.logaddexp(0, -wxy))[:, np.newaxis]
                dphi = -c * (w[:, y].T - w[:, t].T)  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            dmu = self._get_dmu(domega, dphi, z) / n  # (M,d)
            dgamma = self._get_dgamma(gamma, domega, dphi, z) / n  # (M,d)
            dalpha = self._get_dalpha(domega, dphi, z, mu, gamma) / n  # (M,)
        else:
            if self.loss == 'hinge':
                wxy = y * wx
                # compute gradients of \mu
                dphi = -y[wxy < 1, np.newaxis] * w  # (N,2D) (gradient of \Phi(x))
                domega = self._get_domega(x[wxy < 1], phi[wxy < 1])  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'l1':
                wxy = np.sign(wx - y)[:, np.newaxis]
                dphi = wxy * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'l2':
                wxy = (wx - y)[:, np.newaxis]
                dphi = wxy * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'logit':
                wxy = y * wx
                c = (-y * np.exp(-wxy - np.logaddexp(0, -wxy)))[:, np.newaxis]
                dphi = c * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'eps_insensitive':
                wxy = np.abs(y - wx) > self.eps
                c = np.sign(wx - y)[:, np.newaxis]
                d = c * phi
                dphi = c * w  # (N,2D)
                domega = self._get_domega(x[wxy], phi[wxy])  # (N,2D,d) (gradient of \omega)

            dmu = self._get_dmu(domega, dphi, z) / n  # (M,d)
            dgamma = self._get_dgamma(gamma, domega, dphi, z) / n  # (M,d)
            dalpha = self._get_dalpha(domega, dphi, z, mu, gamma) / n  # (M,)

        return dmu, dgamma, dalpha

    def get_grad_alpha(self, x, y, *args, **kwargs):
        n = x.shape[0]  # batch size
        z = kwargs['z'] if 'z' in kwargs else self.z  # (D,)
        w = kwargs['w'] if 'w' in kwargs else self.w  # (2D,C)
        mu = kwargs['mu'] if 'mu' in kwargs else self.mu  # (M,d)
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_  # (M,d)
        alpha = kwargs['alpha'] if 'alpha' in kwargs else self.alpha  # (M,)

        phi = kwargs['phi'] if 'phi' in kwargs else self._get_phi(x, **kwargs)  # (N,2D)
        wx = kwargs['wx'] if 'wx' in kwargs else phi.dot(w)  # (N,C)

        dalpha = np.zeros(alpha.shape)

        if self.num_classes > 2:
            wxy, t = self._get_wxy(wx, y)
            if self.loss == 'hinge':
                dphi = -w[:, y[wxy < 1]].T + w[:, t[wxy < 1]].T  # (N,2D)
                domega = self._get_domega(x[wxy < 1], phi[wxy < 1])  # (N,2D,d) (gradient of \omega)
            else:  # logit loss
                c = np.exp(-wxy - np.logaddexp(0, -wxy))[:, np.newaxis]
                dphi = -c * (w[:, y].T - w[:, t].T)  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)

            dalpha += self._get_dalpha(domega, dphi, z, mu, gamma) / n  # (M,)
        else:
            if self.loss == 'hinge':
                wxy = y * wx

                # compute gradients of \mu
                dphi = -y[wxy < 1, np.newaxis] * w  # (N,2D) (gradient of \Phi(x))
                domega = self._get_domega(x[wxy < 1], phi[wxy < 1])  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'l1':
                wxy = np.sign(wx - y)[:, np.newaxis]
                dphi = wxy * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'l2':
                wxy = (wx - y)[:, np.newaxis]
                dphi = wxy * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'logit':
                wxy = y * wx
                c = (-y * np.exp(-wxy - np.logaddexp(0, -wxy)))[:, np.newaxis]
                dphi = c * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'eps_insensitive':
                wxy = np.abs(y - wx) > self.eps
                c = np.sign(wx - y)[:, np.newaxis]
                dphi = c[wxy] * w  # (N,2D)
                domega = self._get_domega(x[wxy], phi[wxy])  # (N,2D,d) (gradient of \omega)

            dalpha += self._get_dalpha(domega, dphi, z, mu, gamma) / n  # (M,)

        return dalpha

    def _get_dw(self, x, y, *args, **kwargs):
        n = x.shape[0]  # batch size
        w = kwargs['w'] if 'w' in kwargs else self.w  # (2D,C)

        phi = kwargs['phi'] if 'phi' in kwargs else self._get_phi(x, **kwargs)  # (N,2D)
        wx = kwargs['wx'] if 'wx' in kwargs else phi.dot(w)  # (N,C)

        dw = self.lbd * w  # (2D,C)

        if self.num_classes > 2:
            wxy, t = self._get_wxy(wx, y)
            if self.loss == 'hinge':
                d = (wxy[:, np.newaxis] < 1) * phi  # (N,2D)
            else:  # logit loss
                c = np.exp(-wxy - np.logaddexp(0, -wxy))[:, np.newaxis]
                d = c * phi
            for i in range(self.num_classes):
                dw[:, i] += -d[y == i].sum(axis=0) / n
                dw[:, i] += d[t == i].sum(axis=0) / n
        else:
            if self.loss == 'hinge':
                wxy = y * wx
                dw += np.sum(-y[wxy < 1, np.newaxis] * phi[wxy < 1], axis=0) / n
            elif self.loss == 'l1':
                wxy = np.sign(wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
            elif self.loss == 'l2':
                wxy = (wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
            elif self.loss == 'logit':
                wxy = y * wx
                c = (-y * np.exp(-wxy - np.logaddexp(0, -wxy)))[:, np.newaxis]
                dw += np.mean(c * phi, axis=0)
            elif self.loss == 'eps_insensitive':
                wxy = np.abs(y - wx) > self.eps
                c = np.sign(wx - y)[:, np.newaxis]
                d = c * phi
                dw += d[wxy].sum(axis=0) / n

        return dw

    def _get_domega(self, x, phi):
        n = x.shape[0]  # batch size
        domega = np.zeros([n, 2 * self.D, self.num_features])  # (N,2D,d)
        coswx, sinwx = phi[:, :self.D], phi[:, self.D:]  # (N,D)
        domega[:, :self.D, :] = np.einsum("mn,md->mdn", -x, sinwx)
        domega[:, self.D:, :] = np.einsum("mn,md->mdn", x, coswx)
        return domega

    def _get_dmu(self, domega, dphi, z):
        # dmu = np.zeros([self.num_kernels, self.d_])   # (M,d)
        return np.einsum("mdn,md,dk->kn", domega, dphi, np.vstack((z, z)))

    def _get_dgamma(self, gamma, domega, dphi, z):
        # dgamma = np.zeros([self.num_kernels, self.d_])      # (M,d)
        return np.einsum("mdn,md,nd,dk,kn->kn", domega, dphi, np.hstack([self.e, self.e]),
                         np.vstack([z, z]), np.exp(gamma))

    def _get_dalpha(self, domega, dphi, z, mu, gamma):
        dalpha = np.zeros(self.num_kernels)  # (M,)
        omega = self._get_omega(z=z, mu=mu, gamma=gamma)  # (d,D)

        for m in range(self.num_kernels):
            tmp = z[:, m] * (mu[[m]].T + np.exp(gamma[[m]].T) * self.e)  # (d,D)
            tmp = np.hstack([tmp, tmp])
            dalpha[m] += np.einsum("mdn,nd,md", domega, tmp, dphi) / self.temperature
            tmp = z[:, m].T * omega  # (d,D)
            tmp = np.hstack([tmp, tmp])
            dalpha[m] -= np.einsum("mdn,nd,md", domega, tmp, dphi) / self.temperature
        return dalpha

    def _get_omega(self, **kwargs):
        mu = kwargs['mu'] if 'mu' in kwargs else self.mu
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_
        z = kwargs['z'] if 'z' in kwargs else self.z  # (D,M)

        omega = mu.T.dot(z.T) + np.exp(gamma.T).dot(z.T) * self.e  # (d,D)

        return omega

    def _get_phi(self, x, **kwargs):
        omega = kwargs['omega'] if 'omega' in kwargs else self._get_omega(**kwargs)
        phi = np.zeros([x.shape[0], 2 * self.D])  # (N,2D)
        xo = x.dot(omega)
        phi[:, :self.D] = np.cos(xo)
        phi[:, self.D:] = np.sin(xo)
        return phi

    def _fit_online(self, x, y):
        c = 0
        c_w = 0
        s_w, t_w = np.zeros(self.w.shape), np.zeros(self.w.shape)
        s_mu, t_mu = np.zeros(self.mu.shape), np.zeros(self.mu.shape)
        s_gamma, t_gamma = np.zeros(self.gamma_.shape), np.zeros(self.gamma_.shape)
        s_alpha, t_alpha = np.zeros(self.alpha.shape), np.zeros(self.alpha.shape)

        y0 = self._decode_labels(y)
        mistake = 0.0

        for t in range(x.shape[0]):
            phi = self._get_phi(x[[t]])

            wx = phi.dot(self.w)  # (x,)
            if self.task == 'classification':
                if self.num_classes == 2:
                    y_pred = self._decode_labels(np.uint8(wx >= 0))[0]
                else:
                    y_pred = self._decode_labels(np.argmax(wx))
                mistake += (y_pred != y0[t])
            else:
                mistake += (wx[0] - y0[t]) ** 2

            if self.alternative_update:
                dmu, dgamma, dalpha = self.get_grad_mu_gamma_alpha(x[[t]], y[[t]],
                                                                   phi=phi, wx=wx)
                if not self.adam_update:
                    s_mu = self.momentum * s_mu + self.learning_rate_mu * dmu
                    self.mu -= s_mu
                    s_gamma = self.momentum * s_gamma + self.learning_rate_gamma * dgamma
                    self.gamma_ -= s_gamma
                    s_alpha = self.momentum * s_alpha + self.learning_rate_alpha * dalpha
                    self.alpha -= s_alpha
                else:
                    # adam update
                    c += 1
                    dmu, s_mu, t_mu = self._get_adam_update(
                        c, s_mu, t_mu, dmu, self.learning_rate_mu)
                    self.mu -= dmu
                    dgamma, s_gamma, t_gamma = self._get_adam_update(
                        c, s_gamma, t_gamma, dgamma, self.learning_rate_gamma)
                    self.gamma_ -= dgamma
                    dalpha, s_alpha, t_alpha = self._get_adam_update(
                        c, s_alpha, t_alpha, dalpha, self.learning_rate_alpha)
                    self.alpha -= dalpha

                self._update_z()

                for i in range(self.num_nested_epochs):
                    dw = self._get_dw(x[[t]], y[[t]])

                    if not self.adam_update:
                        s_w = self.momentum * s_w + self.learning_rate * dw
                        self.w -= s_w
                    else:
                        # adam update
                        c_w += 1
                        dw, s_w, t_w = self._get_adam_update(
                            c_w, s_w, t_w, dw, self.learning_rate)
                        self.w -= dw

            else:  # if not self.alternative_update:
                # compute gradients
                # dw, dmu, dgamma = self.get_grad(x[[t]], y[[t]], phi=phi, wx=wx)
                # dalpha = self.get_grad_alpha(x[[t]], y[[t]], phi=phi, wx=wx)

                dw, dmu, dgamma, dalpha = self.get_grad_all(x[[t]], y[[t]], phi=phi, wx=wx)

                # update parameters
                if not self.adam_update:
                    s_w = self.momentum * s_w + self.learning_rate * dw
                    self.w -= s_w
                    s_mu = self.momentum * s_mu + self.learning_rate_mu * dmu
                    self.mu -= s_mu
                    s_gamma = self.momentum * s_gamma \
                              + self.learning_rate_gamma * dgamma
                    self.gamma_ -= s_gamma
                    s_alpha = self.momentum * s_alpha \
                              + self.learning_rate_alpha * dalpha
                    self.alpha -= s_alpha
                else:
                    # adam update
                    c += 1
                    dw, s_w, t_w = self._get_adam_update(
                        c, s_w, t_w, dw, self.learning_rate)
                    self.w -= dw
                    dmu, s_mu, t_mu = self._get_adam_update(
                        c, s_mu, t_mu, dmu, self.learning_rate_mu)
                    self.mu -= dmu
                    dgamma, s_gamma, t_gamma = self._get_adam_update(
                        c, s_gamma, t_gamma, dgamma, self.learning_rate_gamma)
                    self.gamma_ -= dgamma
                    dalpha, s_alpha, t_alpha = self._get_adam_update(
                        c, s_alpha, t_alpha, dalpha, self.learning_rate_alpha)
                    self.alpha -= dalpha
                self._update_z()

        self.mistake = mistake / x.shape[0]

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):

        if self.mode == 'online':
            self._fit_online(x, y)
        else:  # batch mode
            c = 0
            c_alpha = 0
            c_w = 0
            s_w, t_w = np.zeros(self.w.shape), np.zeros(self.w.shape)
            s_mu, t_mu = np.zeros(self.mu.shape), np.zeros(self.mu.shape)
            s_gamma, t_gamma = np.zeros(self.gamma_.shape), np.zeros(self.gamma_.shape)
            s_alpha, t_alpha = np.zeros(self.alpha.shape), np.zeros(self.alpha.shape)

            batches = make_batches(x.shape[0], self.batch_size)

            while (self.epoch < self.num_epochs) and (not self.stop_training):
                epoch_logs = {}
                callbacks.on_epoch_begin(self.epoch)

                if self.alternative_update:
                    for batch_idx, (batch_start, batch_end) in enumerate(batches):
                        batch_logs = {'batch': batch_idx,
                                      'size': batch_end - batch_start}
                        callbacks.on_batch_begin(batch_idx, batch_logs)

                        x_batch = x[batch_start:batch_end]
                        y_batch = y[batch_start:batch_end]

                        dmu, dgamma, dalpha = self.get_grad_mu_gamma_alpha(x_batch, y_batch)

                        if not self.adam_update:
                            s_mu = self.momentum * s_mu + self.learning_rate_mu * dmu
                            self.mu -= s_mu
                            s_gamma = self.momentum * s_gamma + self.learning_rate_gamma * dgamma
                            self.gamma_ -= s_gamma
                            s_alpha = self.momentum * s_alpha + self.learning_rate_alpha * dalpha
                            self.alpha -= s_alpha
                        else:
                            # adam update
                            c += 1
                            dmu, s_mu, t_mu = self._get_adam_update(
                                c, s_mu, t_mu, dmu, self.learning_rate_mu)
                            self.mu -= dmu
                            dgamma, s_gamma, t_gamma = self._get_adam_update(
                                c, s_gamma, t_gamma, dgamma, self.learning_rate_gamma)
                            self.gamma_ -= dgamma
                            dalpha, s_alpha, t_alpha = self._get_adam_update(
                                c, s_alpha, t_alpha, dalpha, self.learning_rate_alpha)
                            self.alpha -= dalpha
                        self._update_z()

                        for i in range(self.num_nested_epochs):
                            dw = self._get_dw(x_batch, y_batch)
                            if not self.adam_update:
                                s_w = self.momentum * s_w + self.learning_rate * dw
                                self.w -= s_w
                            else:
                                # adam update
                                c_w += 1
                                dw, s_w, t_w = self._get_adam_update(
                                    c_w, s_w, t_w, dw, self.learning_rate)
                                self.w -= dw

                        batch_logs.update(self._on_batch_end(x_batch, y_batch))
                        callbacks.on_batch_end(batch_idx, batch_logs)

                else:  # if not self.alternative_update:
                    if self.num_nested_epochs > 0:

                        for i in range(self.num_nested_epochs):
                            for batch_idx, (batch_start, batch_end) in enumerate(batches):
                                x_batch = x[batch_start:batch_end]
                                y_batch = y[batch_start:batch_end]

                                dw, dmu, dgamma = self.get_grad(x_batch, y_batch)

                                if not self.adam_update:
                                    s_w = self.momentum * s_w + self.learning_rate * dw
                                    self.w -= s_w
                                    s_mu = self.momentum * s_mu + self.learning_rate_mu * dmu
                                    self.mu -= s_mu
                                    s_gamma = self.momentum * s_gamma \
                                              + self.learning_rate_gamma * dgamma
                                    self.gamma_ -= s_gamma
                                else:
                                    # adam update
                                    c += 1
                                    dw, s_w, t_w = self._get_adam_update(
                                        c, s_w, t_w, dw, self.learning_rate)
                                    self.w -= dw
                                    dmu, s_mu, t_mu = self._get_adam_update(
                                        c, s_mu, t_mu, dmu, self.learning_rate_mu)
                                    self.mu -= dmu
                                    dgamma, s_gamma, t_gamma = self._get_adam_update(
                                        c, s_gamma, t_gamma, dgamma, self.learning_rate_gamma)
                                    self.gamma_ -= dgamma

                        for batch_idx, (batch_start, batch_end) in enumerate(batches):
                            batch_logs = {'batch': batch_idx,
                                          'size': batch_end - batch_start}
                            callbacks.on_batch_begin(batch_idx, batch_logs)

                            x_batch = x[batch_start:batch_end]
                            y_batch = y[batch_start:batch_end]

                            dalpha = self.get_grad_alpha(x_batch, y_batch)

                            if not self.adam_update:
                                s_alpha = self.momentum * s_alpha \
                                          + self.learning_rate_alpha * dalpha
                                self.alpha -= s_alpha
                            else:
                                c_alpha += 1
                                dalpha, s_alpha, t_alpha = self._get_adam_update(
                                    c_alpha, s_alpha, t_alpha, dalpha, self.learning_rate_alpha)
                                self.alpha -= dalpha
                            self._update_z()

                            batch_logs.update(self._on_batch_end(x_batch, y_batch))
                            callbacks.on_batch_end(batch_idx, batch_logs)

                    else:  # self.num_nested_epochs = 0

                        for batch_idx, (batch_start, batch_end) in enumerate(batches):
                            batch_logs = {'batch': batch_idx,
                                          'size': batch_end - batch_start}
                            callbacks.on_batch_begin(batch_idx, batch_logs)

                            x_batch = x[batch_start:batch_end]
                            y_batch = y[batch_start:batch_end]

                            dw, dmu, dgamma, dalpha = self.get_grad_all(x_batch, y_batch)

                            if not self.adam_update:
                                s_w = self.momentum * s_w + self.learning_rate * dw
                                self.w -= s_w
                                s_mu = self.momentum * s_mu + self.learning_rate_mu * dmu
                                self.mu -= s_mu
                                s_gamma = self.momentum * s_gamma \
                                          + self.learning_rate_gamma * dgamma
                                self.gamma_ -= s_gamma
                                s_alpha = self.momentum * s_alpha \
                                          + self.learning_rate_alpha * dalpha
                                self.alpha -= s_alpha
                            else:
                                # adam update
                                c += 1
                                dw, s_w, t_w = self._get_adam_update(
                                    c, s_w, t_w, dw, self.learning_rate)
                                self.w -= dw
                                dmu, s_mu, t_mu = self._get_adam_update(
                                    c, s_mu, t_mu, dmu, self.learning_rate_mu)
                                self.mu -= dmu
                                dgamma, s_gamma, t_gamma = self._get_adam_update(
                                    c, s_gamma, t_gamma, dgamma, self.learning_rate_gamma)
                                self.gamma_ -= dgamma
                                dalpha, s_alpha, t_alpha = self._get_adam_update(
                                    c, s_alpha, t_alpha, dalpha, self.learning_rate_alpha)
                                self.alpha -= dalpha

                            self._update_z()

                            batch_logs.update(self._on_batch_end(x_batch, y_batch))
                            callbacks.on_batch_end(batch_idx, batch_logs)

                            # end of (if self.num_nested_epochs > 0:)

                # end of (if self.alternative_update:)

                if do_validation:
                    outs = self._on_batch_end(x_valid, self._transform_labels(y_valid))
                    for key, value in outs.items():
                        epoch_logs['val_' + key] = value

                epoch_logs.update({'mu': self.mu[0, 0], 'gamma': np.exp(self.gamma_[0][0])})
                callbacks.on_epoch_end(self.epoch, epoch_logs)
                self._on_epoch_end()

    def _on_epoch_end(self):
        super(KMM, self)._on_epoch_end()
        if self.anneal_rate > 0.0:
            self.temperature = max(self.min_temperature, self.temperature * self.anneal_rate)

    def _get_adam_update(self, c, s, t, d, lr):
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        s = beta1 * s + (1 - beta1) * d
        t = beta2 * t + (1 - beta2) * d * d
        lr1 = lr * np.sqrt(1 - beta2 ** c) / (1 - beta1 ** c)
        return lr1 * (s / (eps + np.sqrt(t))), s, t

    def _roll_params(self):
        return np.concatenate([super(KMM, self)._roll_params(),
                               np.ravel(self.mu.copy()), np.ravel(self.alpha.copy())])

    def _unroll_params(self, w):
        ww = super(KMM, self)._unroll_params(w)
        ww = tuple([ww]) if not isinstance(ww, tuple) else ww
        idx = np.sum([i.size for i in ww], dtype=np.int)
        mu = w[idx:idx + self.mu.size].reshape(self.mu.shape).copy()
        idx += self.mu.size
        alpha = w[idx:idx + self.alpha.size].reshape(self.alpha.shape).copy()
        return ww + (mu, alpha)

    def get_loss(self, x, y, **kwargs):
        if 'mu' not in kwargs:
            kwargs['mu'] = self.mu
        if 'alpha' not in kwargs:
            kwargs['alpha'] = self.alpha
        return super(KMM, self).get_loss(x, y, **kwargs)

    def _get_loss_check_grad(self, w, x, y):
        ww, gamma, mu, alpha = self._unroll_params(w)
        self.z = self._get_z(alpha=alpha)
        return self.get_loss(x, y, w=ww, mu=mu, gamma=gamma, alpha=alpha)

    def _get_grad_check_grad(self, w, x, y):
        ww, gamma, mu, alpha = self._unroll_params(w)
        self.z = self._get_z(alpha=alpha)
        dw, dmu, dgamma = self.get_grad(x, y, w=ww, mu=mu, gamma=gamma, alpha=alpha)
        dalpha = self.get_grad_alpha(x, y, w=ww, mu=mu, gamma=gamma, alpha=alpha)
        return np.concatenate([np.ravel(dw), np.ravel(dgamma), np.ravel(dmu), np.ravel(dalpha)])

    def check_grad_online(self, x, y):
        """Check gradients of the model using data X and label y if available
         """
        self._init()

        if y is not None:
            # encode labels
            y = self._encode_labels(y)

        # initialize weights
        self._init_params(x)

        print("Checking gradient... ", end='')

        s = 0.0
        for t in range(x.shape[0]):
            s += check_grad(self._get_loss_check_grad,
                            self._get_grad_check_grad,
                            self._roll_params(),
                            x[[t]], y[[t]])

            dw, dmu, dgamma = self.get_grad(x[[t]], y[[t]])
            dalpha = self.get_grad_alpha(x[[t]], y[[t]])
            self.w -= self.learning_rate * dw
            self.mu -= self.learning_rate_mu * dmu
            self.gamma_ -= self.learning_rate_gamma * dgamma
            self.alpha -= self.learning_rate_alpha * dalpha

        s /= x.shape[0]
        print("diff = %.8f" % s)
        return s

    def disp_syndata(self, **kwargs):
        t, kt, approx_kt = kwargs['t'], kwargs['kt'], kwargs['approx_kt']
        ax = kwargs['ax']
        ax.plot(t, kt, 'r--', linewidth=3, label='True')
        ax.plot(t, approx_kt, 'c-.', linewidth=3, label='Distro')
        # ax.set_ylim(-0.5, 1)

        xx = t
        yy = np.zeros(xx.shape)
        phi_xxx = self._get_phi(xx[:, np.newaxis])
        phi_yyy = self._get_phi(yy[:, np.newaxis])
        approx_ktt = np.sum(phi_xxx * phi_yyy / self.D, axis=1)
        ax.plot(t, approx_ktt, 'b-', linewidth=3, label='KMM')

    def display(self, param, **kwargs):
        if param == 'syndata':
            self.disp_syndata(**kwargs)
        else:
            raise NotImplementedError

    def get_params(self, deep=True):
        out = super(KMM, self).get_params(deep=deep)
        param_names = KMM._get_param_names()
        out.update(self._get_params(param_names=param_names, deep=deep))
        return out


class KMM0(KMM):
    """Kernel Mixture Models optimized for not learning mean parameter (\mu)
    """

    def _fit_loop(self, x, y,
                  do_validation=False,
                  x_valid=None, y_valid=None,
                  callbacks=None, callback_metrics=None):
        if self.mode == 'online':
            y0 = self._decode_labels(y)
            mistake = 0.0

            for t in range(x.shape[0]):
                phi = self._get_phi(x[[t]])

                wx = phi.dot(self.w)  # (x,)
                if self.task == 'classification':
                    if self.num_classes == 2:
                        y_pred = self._decode_labels(np.uint8(wx >= 0))[0]
                    else:
                        y_pred = self._decode_labels(np.argmax(wx))
                    mistake += (y_pred != y0[t])
                else:
                    mistake += (wx[0] - y0[t]) ** 2

                dw, dgamma = self.get_grad(x[[t]], y[[t]], phi=phi, wx=wx)  # compute gradients
                dalpha = self.get_grad_alpha(x[[t]], y[[t]], phi=phi, wx=wx)

                # update parameters
                self.w -= self.learning_rate * dw
                self.gamma_ -= self.learning_rate_gamma * dgamma
                self.alpha -= self.learning_rate_alpha * dalpha
                self._update_z()

            self.mistake = mistake / x.shape[0]

        else:  # batch mode
            c = 0
            c_alpha = 0
            s_w, t_w = np.zeros(self.w.shape), np.zeros(self.w.shape)
            s_gamma, t_gamma = np.zeros(self.gamma_.shape), np.zeros(self.gamma_.shape)
            s_alpha, t_alpha = np.zeros(self.alpha.shape), np.zeros(self.alpha.shape)

            batches = make_batches(x.shape[0], self.batch_size)

            while self.epoch_ < self.num_epochs:
                epoch_logs = {}
                callbacks.on_epoch_begin(self.epoch_)

                if self.num_nested_epochs > 0:

                    for i in range(self.num_nested_epochs):
                        for batch_idx, (batch_start, batch_end) in enumerate(batches):
                            x_batch = x[batch_start:batch_end]
                            y_batch = y[batch_start:batch_end]

                            dw, dgamma = self.get_grad(x_batch, y_batch)

                            # update
                            # self.w_ -= self.learning_rate * dw
                            # self.mu_ -= self.learning_rate_mu * dmu
                            # self.gamma_ -= self.learning_rate_gamma * dgamma

                            # adam update
                            c += 1
                            dw, s_w, t_w = self._get_adam_update(
                                c, s_w, t_w, dw, self.learning_rate)
                            self.w -= dw
                            dgamma, s_gamma, t_gamma = self._get_adam_update(
                                c, s_gamma, t_gamma, dgamma, self.learning_rate_gamma)
                            self.gamma_ -= dgamma

                    for batch_idx, (batch_start, batch_end) in enumerate(batches):
                        x_batch = x[batch_start:batch_end]
                        y_batch = y[batch_start:batch_end]

                        dalpha = self.get_grad_alpha(x_batch, y_batch)
                        c_alpha += 1
                        dalpha, s_alpha, t_alpha = self._get_adam_update(
                            c_alpha, s_alpha, t_alpha, dalpha, self.learning_rate_alpha)
                        self.alpha -= dalpha
                        self._update_z()

                else:  # num_nested_epochs = 0

                    for batch_idx, (batch_start, batch_end) in enumerate(batches):
                        x_batch = x[batch_start:batch_end]
                        y_batch = y[batch_start:batch_end]

                        dw, dgamma, dalpha = self.get_grad_all(x_batch, y_batch)

                        # update
                        # self.w_ -= self.learning_rate * dw
                        # self.mu_ -= self.learning_rate_mu * dmu
                        # self.gamma_ -= self.learning_rate_gamma * dgamma
                        # self.alpha_ -= self.learning_rate_alpha * dalpha

                        # adam update
                        c += 1
                        dw, s_w, t_w = self._get_adam_update(
                            c, s_w, t_w, dw, self.learning_rate)
                        self.w -= dw
                        dgamma, s_gamma, t_gamma = self._get_adam_update(
                            c, s_gamma, t_gamma, dgamma, self.learning_rate_gamma)
                        self.gamma_ -= dgamma
                        dalpha, s_alpha, t_alpha = self._get_adam_update(
                            c, s_alpha, t_alpha, dalpha, self.learning_rate_alpha)
                        self.alpha -= dalpha
                        self._update_z()

                callbacks.on_epoch_end(self.epoch, epoch_logs)

                self.epoch += 1
                if self.stop_training:
                    self.epoch = self.stop_training
                    break

    def get_grad(self, x, y, *args, **kwargs):
        n = x.shape[0]  # batch size
        z = kwargs['z'] if 'z' in kwargs else self.z  # (D,)
        w = kwargs['w'] if 'w' in kwargs else self.w  # (2D,C)
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_  # (M,d)

        phi = kwargs['phi'] if 'phi' in kwargs else self._get_phi(x, **kwargs)  # (N,2D)
        wx = kwargs['wx'] if 'wx' in kwargs else phi.dot(w)  # (N,C)

        dw = self.lbd * w  # (2D,C)

        if self.num_classes > 2:
            wxy, t = self._get_wxy(wx, y)
            if self.loss == 'hinge':
                d = (wxy[:, np.newaxis] < 1) * phi  # (N,2D)
                dphi = -w[:, y[wxy < 1]].T + w[:, t[wxy < 1]].T  # (N,2D)
                domega = self._get_domega(x[wxy < 1], phi[wxy < 1])  # (N,2D,d) (gradient of \omega)
            else:  # logit loss
                c = np.exp(-wxy - np.logaddexp(0, -wxy))[:, np.newaxis]
                d = c * phi
                dphi = -c * (w[:, y].T - w[:, t].T)  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            for i in range(self.num_classes):
                dw[:, i] += -d[y == i].sum(axis=0) / n
                dw[:, i] += d[t == i].sum(axis=0) / n
            dgamma = self._get_dgamma(gamma, domega, dphi, z) / n  # (M,d)
        else:
            if self.loss == 'hinge':
                wxy = y * wx
                dw += np.sum(-y[wxy < 1, np.newaxis] * phi[wxy < 1], axis=0) / n

                # compute gradients of \mu
                dphi = -y[wxy < 1, np.newaxis] * w  # (N,2D) (gradient of \Phi(x))
                domega = self._get_domega(x[wxy < 1], phi[wxy < 1])  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'l1':
                wxy = np.sign(wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'l2':
                wxy = (wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'logit':
                wxy = y * wx
                c = (-y * np.exp(-wxy - np.logaddexp(0, -wxy)))[:, np.newaxis]
                dw += np.mean(c * phi, axis=0)
                dphi = c * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'eps_insensitive':
                wxy = np.abs(y - wx) > self.eps
                c = np.sign(wx - y)[:, np.newaxis]
                d = c * phi
                dw += d[wxy].sum(axis=0) / n
                dphi = c * w  # (N,2D)
                domega = self._get_domega(x[wxy], phi[wxy])  # (N,2D,d) (gradient of \omega)

            dgamma = self._get_dgamma(gamma, domega, dphi, z) / n  # (M,d)

        return dw, dgamma

    def get_grad_all(self, x, y, *args, **kwargs):
        n = x.shape[0]  # batch size
        z = kwargs['z'] if 'z' in kwargs else self.z  # (D,)
        w = kwargs['w'] if 'w' in kwargs else self.w  # (2D,C)
        mu = kwargs['mu'] if 'mu' in kwargs else self.mu  # (M,d)
        gamma = kwargs['gamma'] if 'gamma' in kwargs else self.gamma_  # (M,d)

        phi = kwargs['phi'] if 'phi' in kwargs else self._get_phi(x, **kwargs)  # (N,2D)
        wx = kwargs['wx'] if 'wx' in kwargs else phi.dot(w)  # (N,C)

        dw = self.lbd * w  # (2D,C)

        if self.num_classes > 2:
            wxy, t = self._get_wxy(wx, y)
            if self.loss == 'hinge':
                d = (wxy[:, np.newaxis] < 1) * phi  # (N,2D)
                dphi = -w[:, y[wxy < 1]].T + w[:, t[wxy < 1]].T  # (N,2D)
                domega = self._get_domega(x[wxy < 1], phi[wxy < 1])  # (N,2D,d) (gradient of \omega)
            else:  # logit loss
                c = np.exp(-wxy - np.logaddexp(0, -wxy))[:, np.newaxis]
                d = c * phi
                dphi = -c * (w[:, y].T - w[:, t].T)  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            for i in range(self.num_classes):
                dw[:, i] += -d[y == i].sum(axis=0) / n
                dw[:, i] += d[t == i].sum(axis=0) / n
            dgamma = self._get_dgamma(gamma, domega, dphi, z) / n  # (M,d)
            dalpha = self._get_dalpha(domega, dphi, z, mu, gamma) / n  # (M,)
        else:
            if self.loss == 'hinge':
                wxy = y * wx
                dw += np.sum(-y[wxy < 1, np.newaxis] * phi[wxy < 1], axis=0) / n

                # compute gradients of \mu
                dphi = -y[wxy < 1, np.newaxis] * w  # (N,2D) (gradient of \Phi(x))
                domega = self._get_domega(x[wxy < 1], phi[wxy < 1])  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'l1':
                wxy = np.sign(wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'l2':
                wxy = (wx - y)[:, np.newaxis]
                dw += (wxy * phi).mean(axis=0)
                dphi = wxy * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'logit':
                wxy = y * wx
                c = (-y * np.exp(-wxy - np.logaddexp(0, -wxy)))[:, np.newaxis]
                dw += np.mean(c * phi, axis=0)
                dphi = c * w  # (N,2D)
                domega = self._get_domega(x, phi)  # (N,2D,d) (gradient of \omega)
            elif self.loss == 'eps_insensitive':
                wxy = np.abs(y - wx) > self.eps
                c = np.sign(wx - y)[:, np.newaxis]
                d = c * phi
                dw += d[wxy].sum(axis=0) / n
                dphi = c * w  # (N,2D)
                domega = self._get_domega(x[wxy], phi[wxy])  # (N,2D,d) (gradient of \omega)

            dgamma = self._get_dgamma(gamma, domega, dphi, z) / n  # (M,d)
            dalpha = self._get_dalpha(domega, dphi, z, mu, gamma) / n  # (M,)

        return dw, dgamma, dalpha
