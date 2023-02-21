import numpy as np
import pandas as pd

from scipy.special import expit
from scipy.stats import zscore

from compcate.dgp.data_utils import BASE_T


EPS = 0.001


def sim_discrete_time_survival_treatment_setup(
    n,
    d: int = 10,
    t_max: int = 30,
    covariate_model=None,
    covariate_model_params: dict = None,
    treatment_assn_model=None,
    treatment_assn_model_params: dict = None,
    event_model=None,
    event_model_params: dict = None,
    competing_model=None,
    competing_model_params: dict = None,
    seed: int = 42,
    return_long: bool = True,
    return_short: bool = False,
    X_given=None,
):
    """
    Generic function to flexibly simulate a survival (discrete time) treatment setup

    Parameters
    ----------
    n: int
        Number of observations to generate
    d: int
        dimension of X to generate
    covariate_model:
        Model to generate covariates. Default: multivariate normal
    covariate_model_params: dict
        Additional parameters to pass to covariate model
    treatment_assn_model:
        Model to generate propensity scores
    treatment_assn_model_params:
        Additional parameters to pass to propensity model
    event_model:
        Model to generate events of interest
    event_model_params:
        Additional parameters to pass to event model
    competing_model:
        Model generating censoring times
    competing_model_params:
        Additional parameters to pass to censoring model
    seed: int
        Seed
    return_long: bool, default True
        Whether to return data in long format
    return_short: bool, default False
        Whether to return data also in short format

    Returns
    -------
        data_long, data_short
    """
    # set defaults
    if covariate_model is None:
        covariate_model = normal_covariate_model
    else:
        _check_is_callable(covariate_model, "covariate_model")

    if covariate_model_params is None:
        covariate_model_params = {}

    if treatment_assn_model is None:
        treatment_assn_model = propensity_constant
    else:
        _check_is_callable(treatment_assn_model, "treatment_assn_model")

    if treatment_assn_model_params is None:
        treatment_assn_model_params = {}

    if event_model is None:
        event_model = constant_event_model
    else:
        _check_is_callable(event_model, "event_model")

    if event_model_params is None:
        event_model_params = {}

    if competing_model is None:
        competing_model = constant_event_model
    else:
        _check_is_callable(competing_model, "competing_model")

    if competing_model_params is None:
        competing_model_params = {}

    np.random.seed(seed)

    # generate covariates
    if X_given is None:
        X = covariate_model(n=n, d=d, **covariate_model_params)
    else:
        X = X_given

    # assign treatments
    a = treatment_assn_model(X, **treatment_assn_model_params)

    # initialize empty objects for keeping event times
    t_short = np.ones(n) * (t_max + 1)
    delta_short = np.zeros(n)

    # construct initial variables
    idx_atrisk_comp = np.ones(n, dtype=bool)
    idx_atrisk_event = np.ones(n, dtype=bool)
    X_long_c = X.copy()

    if a is None:
        a_long_c = None
        a_long_t = None
    else:
        a_long_c = a.copy()

    # create data by sampling discrete counting process
    for t in range(BASE_T, t_max + BASE_T):
        # determine samples at risk
        if t > BASE_T:
            # no event, not censored, and alive at previous step
            idx_atrisk_comp = (n_t_t == 0) & idx_atrisk_event

            if np.sum(idx_atrisk_comp) == 0:
                # if no more survivors; stop loop
                break

        # sample competing time (for all indivuals only for convenience)
        n_c_t = competing_model(X=X, t=t, a=a, **competing_model_params)

        idx_atrisk_event = (n_c_t == 0) & idx_atrisk_comp

        if np.sum(idx_atrisk_event) == 0:
            # if no more survivors; stop loop
            censored_t = idx_atrisk_comp & (n_c_t == 1)
            delta_short[censored_t] = 0
            t_short[censored_t] = t
            break

        # sample event time (for all indivuals only for convenience)
        n_t_t = event_model(X=X, t=t, a=a, **event_model_params)

        if t == BASE_T:
            n_c = n_c_t
            time_stamps_c = t * np.ones(n)

            n_t = n_t_t[idx_atrisk_event]
            time_stamps_t = t * np.ones(len(n_t))
            X_long_t = X[idx_atrisk_event, :].copy()

            if a is not None:
                a_long_t = a[idx_atrisk_event]

        else:
            n_c = np.concatenate([n_c, n_c_t[idx_atrisk_comp]], axis=0)
            X_long_c = np.concatenate([X_long_c, X[idx_atrisk_comp, :]], axis=0)
            time_stamps_c = np.concatenate(
                [time_stamps_c, t * np.ones(np.sum(idx_atrisk_comp))]
            )

            n_t = np.concatenate([n_t, n_t_t[idx_atrisk_event]], axis=0)
            X_long_t = np.concatenate([X_long_t, X[idx_atrisk_event, :]], axis=0)
            time_stamps_t = np.concatenate(
                [time_stamps_t, t * np.ones(np.sum(idx_atrisk_event))]
            )
            if a is not None:
                a_long_c = np.concatenate([a_long_c, a[idx_atrisk_comp]], axis=0)
                a_long_t = np.concatenate([a_long_t, a[idx_atrisk_event]], axis=0)

        # record short data format: time stamp and event indicator
        events_t = idx_atrisk_event & (n_t_t == 1)
        censored_t = idx_atrisk_comp & (n_c_t == 1)

        t_short[events_t | censored_t] = t
        delta_short[events_t | censored_t] = events_t[events_t | censored_t].astype(int)

    # compile data
    data_short = X, a, t_short, delta_short
    data_long = (
        X_long_c,
        X_long_t,
        a_long_c,
        a_long_t,
        n_c,
        n_t,
        time_stamps_c,
        time_stamps_t,
    )

    if return_short:
        if return_long:
            return data_long, data_short
        else:
            return data_short
    else:
        return data_long


def _check_is_callable(input, name: str = ""):
    if callable(input):
        pass
    else:
        raise ValueError(
            "Input {} needs to be a callable function so it can "
            "be used to create simulation.".format(name)
        )


# simulation models -------------------------------------------------------
def propensity_constant(X, return_p=False, xi: float = 0.5, support_covs=None):
    # samples from constant propensity score model
    p = xi * np.ones(X.shape[0])
    if return_p:
        return np.random.binomial(1, p=p), p
    else:
        return np.random.binomial(1, p=p)


def propensity_linear(
    X,
    return_p=False,
    xi: float = 0.5,
    center: bool = False,
    support_covs=None,
    exponent=1,
):
    if support_covs is None:
        support_covs = [0, 1]

    z = zscore(
        np.average(X[:, support_covs] ** exponent, axis=1)
    )  # original experiment did not
    # have zscore

    if center:
        prop = expit(xi * (z - np.mean(z)))  # original experiment had only this
    else:
        prop = expit(xi * z)

    if return_p:
        return np.random.binomial(1, p=prop), prop
    else:
        return np.random.binomial(1, p=prop)


def constant_event_model(X, t, a=None, xi=0.05, xi_a=0.05, return_p=False):
    # model events with constant probability
    if a is not None:
        p = xi * np.ones(X.shape[0]) + xi_a * a
    else:
        p = xi * np.ones(X.shape[0])

    if return_p:
        return np.random.binomial(1, p=p), p
    else:
        return np.random.binomial(1, p=p)


def binary_hazard_model(
    X,
    t,
    a=None,
    p_group_0=0.01,
    p_group_1=0.1,
    p_a_group_0=0,
    p_a_group_1=0,
    t_effect=None,
    support_cov=0,
    support_cov_a=0,
    return_p=False,
    counterfactual_sep=False,
):
    p = p_group_0 * (X[:, support_cov] <= 0) + p_group_1 * (X[:, support_cov] > 0)
    if (a is not None) and (not counterfactual_sep):
        p += p_a_group_0 * (X[:, support_cov_a] <= 0) * (a == 1) + p_a_group_1 * (
            X[:, support_cov_a] > 0
        ) * (a == 1)
    if t_effect is not None:
        p += t_effect * t

    if return_p:
        return np.random.binomial(1, p=p), p
    else:
        return np.random.binomial(1, p=p)


def competing_model_linear(
    X, t, a, return_p=False, xi_a: float = 0.5, support_covs=None, xi=0.1
):
    if support_covs is None:
        support_covs = [0, 1]

    z = zscore(np.average(X[:, support_covs], axis=1))  # original experiment did not
    # have zscore

    prop = (
        expit(xi_a * (1 - a) * z + np.log(xi)) if t == 1 else xi / t * np.ones_like(a)
    )

    if return_p:
        return np.random.binomial(1, p=prop), prop
    else:
        return np.random.binomial(1, p=prop)


def binary_covariate_model(n, d, p=None, cov_p: float = None):
    if p is None:
        if cov_p is None:
            X = np.random.binomial(1, p=0.5, size=(n, d))
        else:
            X = np.zeros((n, d))
            X[:, 0] = np.random.binomial(1, p=0.5, size=n)
            for idx in range(1, d):
                X[:, idx] = np.random.binomial(
                    1,
                    p=0.5 + cov_p * X[:, (idx - 1)] - cov_p * (1 - X[:, (idx - 1)]),
                    size=n,
                )
    else:
        X = np.transpose(np.random.binomial(1, p=p, size=(d, n)))
    return X


def normal_covariate_model(n, d, rho=0.3, var=1):
    # samples multivariate normal covariates with rho-correlated components
    mean_vec = np.zeros(d)
    Sigma_x = (np.ones([d, d]) * rho + np.identity(d) * (1 - rho)) * var
    X = np.random.multivariate_normal(mean_vec, Sigma_x, n)
    return X
