from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from compcate.dgp.data_utils import short_to_long, _get_values_only, BASE_T

import numpy as np
import pandas as pd

BASE_LR_PARAMS = {}

EPS = 0.001


class TSepHazardModel(BaseEstimator, RegressorMixin):
    """
    A T-learner for survival treatment effect estimation in the presence of competing risks that
    operates by estimating a separate hazard model for every time_step
    """

    def __init__(
        self,
        model_t=None,
        model_c=None,
        model_t_params: dict = None,
        model_c_params: dict = None,
        fit_comp: bool = True,
        est_weight: str = None,
        model_weight=None,
        weight_model_params=None,
        est_prop: bool = False,
        model_propensity=None,
        verbose: bool = False,
    ):
        self.model_t = model_t
        self.model_c = model_c
        self.model_t_params = model_t_params
        self.model_c_params = model_c_params
        self.fit_comp = fit_comp
        self.est_weight = est_weight
        self.model_weight = model_weight
        self.weight_model_params = weight_model_params
        self.est_prop = est_prop
        self.model_propensity = model_propensity
        self.verbose = verbose

    def _prepare_fit(self, time_stamps_c, time_stamps_t):
        if self.model_t is None:
            self.model_t = TLearner

        if self.model_c is None:
            self.model_c = TLearner

        if self.model_weight is None:
            self.model_weight = TLearner

        if self.model_propensity is None:
            self.model_propensity = LogisticRegression(C=100)  # unrestricted LR

        if self.model_c_params is None:
            self.model_c_params = {}

        if self.model_t_params is None:
            self.model_t_params = {}

        if self.weight_model_params is None:
            self.weight_model_params = {}

        self.max_t_ = np.max(time_stamps_t)
        self.t_levels_ = np.unique(time_stamps_t)

        self._model_t = {t: self.model_t(**self.model_t_params) for t in self.t_levels_}

        self._model_c = {
            t: self.model_c(**self.model_c_params)
            if t in np.unique(time_stamps_c)
            else TZeroPredictor()
            for t in self.t_levels_
        }

        if self.est_weight not in [
            None,
            "cause-specific",
            "risk-eliminated",
            "separable",
        ]:
            raise ValueError("Unknown weighting type")

        if self.est_weight is not None:
            self.fit_comp = True

    def fit(
        self, X, t, delta, a, sample_weight_control=None, sample_weight_treated=None
    ):
        # do some initial checks
        X = _get_values_only(X)

        (
            X_long_c,
            X_long_t,
            a_long_c,
            a_long_t,
            n_c,
            n_t,
            time_stamps_c,
            time_stamps_t,
            weights_0,
            weights_1,
        ) = short_to_long(
            X=X,
            t=t,
            delta=delta,
            a=a,
            weights_by_t_control=sample_weight_control,
            weights_by_t_treated=sample_weight_treated,
        )

        # prepare models
        self._prepare_fit(time_stamps_t, time_stamps_c)

        if self.est_weight is not None and self.est_prop:
            self.model_propensity.fit(X, a)

        # fit hazard model by predicting n_c for every time step separately
        for stamp in self.t_levels_:
            if self.verbose:
                print("Fitting time-stamp: {}".format(stamp))

            if self.fit_comp:
                # fit model for competing risk------------------------------------------------

                n_c_t = n_c[time_stamps_c == stamp]  # identify samples at risk

                # check: are enough unique events to fit normal ML model?
                if len(np.unique(n_c[(time_stamps_c == stamp) & (a_long_c == 1)])) < 2:
                    self._model_c[stamp].update_po_estimator(
                        new_estimator=ZeroPredictor(), po=1
                    )
                if len(np.unique(n_c[(time_stamps_c == stamp) & (a_long_c == 0)])) < 2:
                    self._model_c[stamp].update_po_estimator(
                        new_estimator=ZeroPredictor(), po=0
                    )

                # fit model
                self._model_c[stamp].fit(
                    X_long_c[time_stamps_c == stamp, :],
                    n_c_t,
                    a_long_c[time_stamps_c == stamp],
                )

            # fit model for main event ---------------------------------------------

            n_t_t = n_t[time_stamps_t == stamp]  # identify samples at risk

            # check: are enough unique events to fit normal ML model?
            if len(np.unique(n_t[(time_stamps_t == stamp) & (a_long_t == 1)])) < 2:
                self._model_t[stamp].update_po_estimator(
                    new_estimator=ZeroPredictor(), po=1
                )
            if len(np.unique(n_t[(time_stamps_t == stamp) & (a_long_t == 0)])) < 2:
                self._model_t[stamp].update_po_estimator(
                    new_estimator=ZeroPredictor(), po=0
                )

            if weights_0 is not None:
                # use supplied weights
                self._model_t[stamp].fit(
                    X_long_t[time_stamps_t == stamp, :],
                    n_t_t,
                    a_long_t[time_stamps_t == stamp],
                    sample_weight=[
                        weights_0[time_stamps_t == stamp],
                        weights_1[time_stamps_t == stamp],
                    ],
                )
            else:
                if self.est_weight is not None:
                    # compute estimated weights --------------------------------------------
                    # compute competing hazards
                    haz_c_0, haz_c_1 = self.predict_hazards(
                        X_long_t[time_stamps_t == stamp, :],
                        range(BASE_T, int(stamp) + BASE_T),
                        return_comp=True,
                        return_t=False,
                    )
                    # compute appropriate weights from hazard
                    if self.est_weight == "risk-eliminated":
                        weights_est_0 = np.cumprod(1 / ((1.0 - haz_c_0) + EPS), axis=1)[
                            :, -1
                        ]
                        weights_est_1 = np.cumprod(1 / ((1.0 - haz_c_1) + EPS), axis=1)[
                            :, -1
                        ]
                    elif self.est_weight == "separable":
                        weights_est_0 = np.ones(
                            X_long_t[time_stamps_t == stamp, :].shape[0]
                        )
                        weights_est_1 = np.cumprod(
                            (1.0 - haz_c_0) / ((1.0 - haz_c_1) + EPS), axis=1
                        )[:, -1]
                    else:
                        weights_est_0 = np.ones(
                            X_long_t[time_stamps_t == stamp, :].shape[0]
                        )
                        weights_est_1 = np.ones(
                            X_long_t[time_stamps_t == stamp, :].shape[0]
                        )

                    if self.est_prop:
                        # also use estimated propensities
                        weights_est_0 = weights_est_0 * (
                            1
                            / self.model_propensity.predict_proba(
                                X_long_t[time_stamps_t == stamp, :]
                            )[:, 0]
                        )
                        weights_est_1 = weights_est_1 * (
                            1
                            / self.model_propensity.predict_proba(
                                X_long_t[time_stamps_t == stamp, :]
                            )[:, 1]
                        )

                    # fit model with estimated weights
                    self._model_t[stamp].fit(
                        X_long_t[time_stamps_t == stamp, :],
                        n_t_t,
                        a_long_t[time_stamps_t == stamp],
                        sample_weight=[weights_est_0, weights_est_1],
                    )
                else:
                    # fit model without weights
                    self._model_t[stamp].fit(
                        X_long_t[time_stamps_t == stamp, :],
                        n_t_t,
                        a_long_t[time_stamps_t == stamp],
                    )

    def predict_hazard(self, X, t, return_comp=True, return_t=True):
        # predict hazard for time step t

        # check whether there is model corresponding to t
        if t not in self.t_levels_:
            raise ValueError("t needs to be in model.t_levels_")

        X = _get_values_only(X)

        if return_t:  # get hazard for main event
            prob_t_0, prob_t_1 = self._model_t[t].predict_proba(X)

        if return_comp:  # get hazard for competing event
            prob_c_0, prob_c_1 = self._model_c[t].predict_proba(X)

            if return_t:
                return prob_t_0, prob_t_1, prob_c_0, prob_c_1
            else:
                return prob_c_0, prob_c_1
        else:
            if return_t:
                return prob_t_0, prob_t_1
            else:
                return

    def predict_hazards(self, X, ts, return_comp=True, return_t=True):
        # predict hazard for multiple values of t
        if (not return_comp) and (not return_t):
            return

        if return_t:
            hazards_t_0 = np.zeros([X.shape[0], len(ts)])
            hazards_t_1 = np.zeros([X.shape[0], len(ts)])
        if return_comp:
            hazards_c_0 = np.zeros([X.shape[0], len(ts)])
            hazards_c_1 = np.zeros([X.shape[0], len(ts)])

        idx = 0
        for t in ts:
            out_t = self.predict_hazard(
                X, t, return_comp=return_comp, return_t=return_t
            )
            if return_t:
                hazards_t_0[:, idx] = out_t[0]
                hazards_t_1[:, idx] = out_t[1]
                if return_comp:
                    hazards_c_0[:, idx] = out_t[2]
                    hazards_c_1[:, idx] = out_t[3]
            else:
                hazards_c_0[:, idx] = out_t[0]
                hazards_c_1[:, idx] = out_t[1]

            idx = idx + 1

        if return_t:
            if return_comp:
                return (hazards_t_0, hazards_t_1, hazards_c_0, hazards_c_1)
            else:
                return (hazards_t_0, hazards_t_1)
        else:
            return (hazards_c_0, hazards_c_1)

    def compute_effective_sample_size(
        self,
        X,
        t,
        delta,
        a,
        stamp,
        sample_weight_control=None,
        sample_weight_treated=None,
    ):
        # function to compute effective sample size from weights

        X = _get_values_only(X)
        (
            X_long_c,
            X_long_t,
            a_long_c,
            a_long_t,
            n_c,
            n_t,
            time_stamps_c,
            time_stamps_t,
            weights_0,
            weights_1,
        ) = short_to_long(
            X=X,
            t=t,
            delta=delta,
            a=a,
            weights_by_t_control=sample_weight_control,
            weights_by_t_treated=sample_weight_treated,
        )

        if sample_weight_treated is None:
            # compute estimated weights
            if self.est_weight is not None:
                haz_c_0, haz_c_1 = self.predict_hazards(
                    X_long_t[time_stamps_t == stamp, :],
                    range(BASE_T, int(stamp) + BASE_T),
                    return_comp=True,
                    return_t=False,
                )
            else:
                haz_c_0 = None

            if haz_c_0 is not None:
                if self.est_weight == "risk-eliminated":
                    weights_0 = np.cumprod(1 / ((1.0 - haz_c_0) + EPS), axis=1)[:, -1]
                    weights_1 = np.cumprod(1 / ((1.0 - haz_c_1) + EPS), axis=1)[:, -1]
                elif self.est_weight == "separable":
                    weights_0 = np.ones(X_long_t[time_stamps_t == stamp, :].shape[0])
                    weights_1 = np.cumprod(
                        (1.0 - haz_c_0) / ((1.0 - haz_c_1) + EPS), axis=1
                    )[:, -1]
                else:
                    weights_0 = np.ones(X_long_t[time_stamps_t == stamp, :].shape[0])
                    weights_1 = np.ones(X_long_t[time_stamps_t == stamp, :].shape[0])

                if self.est_prop:
                    weights_0 = weights_0 * (
                        1
                        / self.model_propensity.predict_proba(
                            X_long_t[time_stamps_t == stamp, :]
                        )[:, 0]
                    )
                    weights_1 = weights_1 * (
                        1
                        / self.model_propensity.predict_proba(
                            X_long_t[time_stamps_t == stamp, :]
                        )[:, 1]
                    )

            else:
                # no weights
                weights_0 = np.ones(X_long_t[time_stamps_t == stamp, :].shape[0])
                weights_1 = np.ones(X_long_t[time_stamps_t == stamp, :].shape[0])

            # normalize weights
            norm_weights_0 = weights_0[a_long_t[time_stamps_t == stamp] == 0] / np.sum(
                weights_0[a_long_t[time_stamps_t == stamp] == 0]
            )
            norm_weights_1 = weights_1[a_long_t[time_stamps_t == stamp] == 1] / np.sum(
                weights_1[a_long_t[time_stamps_t == stamp] == 1]
            )
        else:
            # normalize weights
            norm_weights_0 = weights_0[
                (time_stamps_t == stamp) & (a_long_t == 0)
            ] / np.sum(weights_0[(time_stamps_t == stamp) & (a_long_t == 0)])
            norm_weights_1 = weights_1[
                (time_stamps_t == stamp) & (a_long_t == 1)
            ] / np.sum(weights_1[(time_stamps_t == stamp) & (a_long_t == 1)])

        # compute effective sample size per treatment group
        n_0 = np.sum(a_long_t[time_stamps_t == stamp] == 0)
        n_1 = np.sum(a_long_t[time_stamps_t == stamp] == 1)

        ess_0 = 1 / (np.sum((norm_weights_0) ** 2))
        ess_1 = 1 / (np.sum((norm_weights_1) ** 2))
        return ess_0, ess_1, n_0, n_1

    def predict_hazards_from_models(self, models, X, ts):
        # predict hazards for all time steps in t from specified models (either object or string)
        if type(models) is str:
            if models == "c":
                models = self._model_c
            elif models == "weight":
                models = self._model_weight
            elif models == "t":
                models = self._model_t
            else:
                raise ValueError("Unknown model.")

        hazards_0 = np.zeros([X.shape[0], len(ts)])
        hazards_1 = np.zeros([X.shape[0], len(ts)])
        idx = 0
        for t in ts:
            haz_0, haz_1 = models[t].predict_proba(X)
            hazards_0[:, idx] = haz_0
            hazards_1[:, idx] = haz_1
            idx += 1
        return hazards_0, hazards_1

    def predict_cause_specific_survivals(self, X, t, risk=False):
        # compute cause-specific survival by time t or risk of main event occuring by time t (
        # associated with total effect; no interventions performed)

        # get all t smaller than t
        ts = range(BASE_T, np.min([self.max_t_, t]).astype(int) + BASE_T)

        if not risk:
            # survival function
            # compute hazards
            hazards_t_0, hazards_t_1, hazards_c_0, hazards_c_1 = self.predict_hazards(
                X, ts
            )
            surv_t_0 = np.cumprod((1.0 - hazards_t_0) * (1.0 - hazards_c_0), axis=1)[
                :, -1
            ]
            surv_t_1 = np.cumprod((1.0 - hazards_t_1) * (1.0 - hazards_c_1), axis=1)[
                :, -1
            ]
        else:
            # total risk (sum over all survival paths)
            # sum over all possible time steps
            surv_t_1 = np.zeros((X.shape[0],))
            surv_t_0 = np.zeros((X.shape[0],))
            for tsub in ts:
                (
                    hazards_t_0,
                    hazards_t_1,
                    hazards_c_0,
                    hazards_c_1,
                ) = self.predict_hazards(X, range(BASE_T, tsub + BASE_T))
                if tsub == BASE_T:
                    surv_t_1 += (hazards_t_1 * (1 - hazards_c_1)).reshape(
                        -1,
                    )
                    surv_t_0 += (hazards_t_0 * (1 - hazards_c_0)).reshape(
                        -1,
                    )
                else:
                    surv_t_1 += (
                        hazards_t_1[:, -1]
                        * np.cumprod((1.0 - hazards_t_1) * (1.0 - hazards_c_1), axis=1)[
                            :, -2
                        ]
                    )
                    surv_t_0 += (
                        hazards_t_0[:, -1]
                        * np.cumprod((1.0 - hazards_t_0) * (1.0 - hazards_c_0), axis=1)[
                            :, -2
                        ]
                    )
        return surv_t_0, surv_t_1

    def predict_risk_eliminated_survival(self, X, t, risk=False):
        # predict survival until time t or risk of main event occuring by time t under elimination
        # of competing risk

        # get all t smaller than t
        ts = range(BASE_T, np.min([self.max_t_, t]).astype(int) + BASE_T)

        if not risk:
            # compute survival through hazards
            hazards_t_0, hazards_t_1 = self.predict_hazards(X, ts, return_comp=False)

            # cumulate into survival
            surv_t_0 = np.cumprod(1.0 - hazards_t_0, axis=1)[:, -1]
            surv_t_1 = np.cumprod(1.0 - hazards_t_1, axis=1)[:, -1]
        else:
            # compute risk of event occuring by time t
            # sum over all possible time steps
            surv_t_1 = np.zeros((X.shape[0],))
            surv_t_0 = np.zeros((X.shape[0],))
            for tsub in ts:
                hazards_t_0, hazards_t_1 = self.predict_hazards(
                    X, range(BASE_T, tsub + BASE_T), return_comp=False
                )
                if tsub == BASE_T:
                    surv_t_1 += hazards_t_1.reshape(
                        -1,
                    )
                    surv_t_0 += hazards_t_0.reshape(
                        -1,
                    )
                else:
                    surv_t_1 += (
                        hazards_t_1[:, -1]
                        * np.cumprod((1.0 - hazards_t_1), axis=1)[:, -2]
                    )
                    surv_t_0 += (
                        hazards_t_0[:, -1]
                        * np.cumprod((1.0 - hazards_t_0), axis=1)[:, -2]
                    )

        return surv_t_0, surv_t_1

    def predict_separable_survival(self, X, t, risk=False):
        # predict survival until time t or risk of main event occuring by time t under separable
        # treatment, where treatment component for competing event is set to 0

        # get all t smaller than t
        ts = range(BASE_T, np.min([self.max_t_, t]).astype(int) + BASE_T)

        # compute hazards
        if not risk:
            hazards_t_0, hazards_t_1, hazards_c_0, hazards_c_1 = self.predict_hazards(
                X, ts
            )

            # cumulate into survival
            surv_t_0 = np.cumprod((1.0 - hazards_t_0) * (1.0 - hazards_c_0), axis=1)[
                :, -1
            ]
            surv_t_1 = np.cumprod((1.0 - hazards_t_1) * (1.0 - hazards_c_0), axis=1)[
                :, -1
            ]
        else:
            # sum over all possible time steps
            surv_t_1 = np.zeros((X.shape[0],))
            surv_t_0 = np.zeros((X.shape[0],))
            for tsub in ts:
                (
                    hazards_t_0,
                    hazards_t_1,
                    hazards_c_0,
                    hazards_c_1,
                ) = self.predict_hazards(X, range(BASE_T, tsub + BASE_T))
                if tsub == BASE_T:
                    surv_t_1 += (hazards_t_1 * (1 - hazards_c_0)).reshape(
                        -1,
                    )
                    surv_t_0 += (hazards_t_0 * (1 - hazards_c_0)).reshape(
                        -1,
                    )
                else:
                    surv_t_1 += (
                        hazards_t_1[:, -1]
                        * np.cumprod((1.0 - hazards_t_1) * (1.0 - hazards_c_0), axis=1)[
                            :, -2
                        ]
                    )
                    surv_t_0 += (
                        hazards_t_0[:, -1]
                        * np.cumprod((1.0 - hazards_t_0) * (1.0 - hazards_c_0), axis=1)[
                            :, -2
                        ]
                    )

        return surv_t_0, surv_t_1

    def predict_rmst(self, X, t, surv_type="risk-eliminated"):
        # predict restricted mean survival time by summing over survival functions

        ts = range(BASE_T, np.min([self.max_t_, t]).astype(int) + BASE_T - 1)
        surv_0, surv_1 = np.ones([X.shape[0], len(ts) + 1]), np.ones(
            [X.shape[0], len(ts) + 1]
        )

        for t_horizon in ts:
            if surv_type == "risk-eliminated":
                surv_t_0, surv_t_1 = self.predict_risk_eliminated_survival(
                    X, t_horizon, risk=False
                )
            elif surv_type == "separable":
                surv_t_0, surv_t_1 = self.predict_separable_survival(
                    X, t_horizon, risk=False
                )
            elif surv_type == "cause-specific":
                surv_t_0, surv_t_1 = self.predict_cause_specific_survivals(
                    X, t_horizon, risk=False
                )
            else:
                raise ValueError("surv-type not recognised.")
            surv_0[:, (t_horizon - BASE_T + 1)] = surv_t_0
            surv_1[:, (t_horizon - BASE_T + 1)] = surv_t_1
        return np.sum(surv_0, axis=1), np.sum(surv_1, axis=1)


class TLearner(BaseEstimator, ClassifierMixin):
    # T-learner class for estimating effects
    def __init__(self, po_estimator=None, **kwargs):
        self.po_estimator = po_estimator
        self.estimator_params = kwargs
        self._po_1 = None
        self._po_0 = None

    def _prepare_self(self):
        if self.po_estimator is None:
            self.po_estimator = LogisticRegression
        if self.estimator_params is None:
            self.estimator_params = {}

        if self._po_0 is None:
            self._po_0 = self.po_estimator(**self.estimator_params)
        if self._po_1 is None:
            self._po_1 = self.po_estimator(**self.estimator_params)

    def update_po_estimator(self, new_estimator, po: int):
        if po == 0:
            self._po_0 = new_estimator
        if po == 1:
            self._po_1 = new_estimator

    def fit(self, X, y, a, sample_weight=None):
        self._prepare_self()
        if sample_weight is not None:
            if type(sample_weight) is list:
                sample_weight_0 = sample_weight[0]
                sample_weight_1 = sample_weight[1]
            else:
                sample_weight_0 = sample_weight
                sample_weight_1 = sample_weight
            self._po_0.fit(
                X[a == 0],
                y[a == 0],
                sample_weight=sample_weight_0[a == 0]
                / (np.sum(sample_weight_0[a == 0]) / np.sum(a == 0)),
            )
            self._po_1.fit(
                X[a == 1],
                y[a == 1],
                sample_weight=sample_weight_1[a == 1]
                / (np.sum(sample_weight_1[a == 1]) / np.sum(a == 1)),
            )
        else:
            self._po_0.fit(X[a == 0], y[a == 0])
            self._po_1.fit(X[a == 1], y[a == 1])

    def predict_proba(self, X, a=None):
        if a is None:
            return self._po_0.predict_proba(X)[:, 1], self._po_1.predict_proba(X)[:, 1]
        else:
            prob_out = np.zeros(X.shape[0])
            prob_out[a == 0] = self._po_0.predict_proba(X[a == 0])[:, 1]
            prob_out[a == 1] = self._po_0.predict_proba(X[a == 1])[:, 1]
            return prob_out


class TZeroPredictor(TLearner):
    # Placeholder predictor used whenever there are no events at a time-step; predicting zero
    # occurance of events (sklearn classifiers fail in this case)
    def fit(self, X, y, a, sample_weight=None):
        pass

    def predict_proba(self, X, a=None):
        if a is None:
            return np.zeros(X.shape[0]), np.zeros(X.shape[0])
        else:
            return np.zeros(X.shape[0])


class ZeroPredictor(BaseEstimator, ClassifierMixin):
    # Placeholder predictor used whenever there are no events at a time-step; predicting zero
    # occurance of events (sklearn classifiers fail in this case)

    def fit(self, X, y, sample_weight=None):
        pass

    def predict_proba(
        self,
        X,
    ):
        return np.zeros((X.shape[0], 2))


class ConstantEstimator(BaseEstimator, ClassifierMixin):
    # estimator for emulating misspecification
    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            # estimate sample average
            self.p_ = np.mean(y)
        else:
            self.p_ = np.sum(sample_weight * y) / np.sum(sample_weight)

    def predict_proba(self, X):
        out = np.zeros((X.shape[0], 2))
        out[:, 0] = 1 - self.p_
        out[:, 1] = self.p_
        return out
