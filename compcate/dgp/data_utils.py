import numpy as np
import pandas as pd

BASE_T = 1


def short_to_long(
    X,
    t,
    delta,
    a=None,
    t_max=None,
    weights_by_t_control=None,
    weights_by_t_treated=None,
):
    # transform short data to long data
    if t_max is None:
        t_max = np.max(t).astype(int) - BASE_T

    # loop through time
    for time in range(BASE_T, t_max + BASE_T):
        # temporal ordering: first do competing risk, then do main event
        idx_atrisk_comp = t >= time
        n_c_t = ((delta[idx_atrisk_comp] == 0) & (t[idx_atrisk_comp] == time)).astype(
            int
        )
        timestamp_c_t = time * np.ones(len(n_c_t))

        idx_atrisk_event = (t > time) | ((delta == 1) & (t == time))
        n_t_t = ((delta[idx_atrisk_event] == 1) & (t[idx_atrisk_event] == time)).astype(
            int
        )
        timestamp_t_t = time * np.ones(len(n_t_t))

        if time == BASE_T:
            X_long_c = X.copy()
            X_long_t = X[idx_atrisk_event, :].copy()
            n_t = n_t_t
            n_c = n_c_t
            time_stamps_t = timestamp_t_t
            time_stamps_c = timestamp_c_t

            if a is not None:  # if there are treatments
                a_long_c = a.copy()
                a_long_t = a[idx_atrisk_event].copy()

            if weights_by_t_control is not None:  # weights should come in nxt matrix
                # weights_long_c = weights_by_t_control[:, 0] not sure if needed
                weights_long_control = weights_by_t_control[idx_atrisk_event, 0]
                weights_long_treated = weights_by_t_treated[idx_atrisk_event, 0]

        else:
            X_long_c = np.concatenate([X_long_c, X[idx_atrisk_comp, :]], axis=0)
            X_long_t = np.concatenate([X_long_t, X[idx_atrisk_event, :]], axis=0)
            n_t = np.concatenate([n_t, n_t_t], axis=0)
            n_c = np.concatenate([n_c, n_c_t], axis=0)
            time_stamps_c = np.concatenate([time_stamps_c, timestamp_c_t], axis=0)
            time_stamps_t = np.concatenate([time_stamps_t, timestamp_t_t], axis=0)

            if a is not None:  # if there are treatments
                a_long_c = np.concatenate([a_long_c, a[idx_atrisk_comp]], axis=0)
                a_long_t = np.concatenate([a_long_t, a[idx_atrisk_event]], axis=0)

            if weights_by_t_control is not None:
                weights_long_control = np.concatenate(
                    [
                        weights_long_control,
                        weights_by_t_control[idx_atrisk_event, time - BASE_T],
                    ]
                )
                weights_long_treated = np.concatenate(
                    [
                        weights_long_treated,
                        weights_by_t_treated[idx_atrisk_event, time - BASE_T],
                    ]
                )

    if a is not None:
        if weights_by_t_control is not None:
            return (
                X_long_c,
                X_long_t,
                a_long_c,
                a_long_t,
                n_c,
                n_t,
                time_stamps_c,
                time_stamps_t,
                weights_long_control,
                weights_long_treated,
            )
        else:
            return (
                X_long_c,
                X_long_t,
                a_long_c,
                a_long_t,
                n_c,
                n_t,
                time_stamps_c,
                time_stamps_t,
                None,
                None,
            )
    else:
        return X_long_c, X_long_t, n_c, n_t, time_stamps_c, time_stamps_t


def get_survivors_from_data(X, t, delta, a, t_horizon, split_by_a=True):
    # get survivors (individuals at risk) for specific time horizon
    (
        X_long_c,
        X_long_t,
        a_long_c,
        a_long_t,
        n_c,
        n_t,
        time_stamps_c,
        time_stamps_t,
        _,
        _,
    ) = short_to_long(X=X, t=t, delta=delta, a=a)

    if split_by_a:
        X_surv_0 = X_long_t[(time_stamps_t == t_horizon) & (a_long_t == 0), :]
        X_surv_1 = X_long_t[(time_stamps_t == t_horizon) & (a_long_t == 1), :]
        return X_surv_0, X_surv_1
    else:
        return X_long_t[(time_stamps_t == t_horizon), :]


def counterfactual_survival_prob_from_discrete_hazard_model(
    X,
    t,
    event_model=None,
    comp_model=None,
    event_model_params=None,
    comp_model_params=None,
    surv_type="cause-specific",
    haz_only=False,
    risk=False,
):
    # compute probabilities from true underlying model
    if event_model is None:
        event_model = constant_event_model
    if comp_model is None:
        comp_model = constant_event_model

    if event_model_params is None:
        event_model_params = {}
    if comp_model_params is None:
        comp_model_params = {}

    if haz_only:
        _, haz_0 = event_model(
            X=X, t=t, a=np.zeros(X.shape[0]), return_p=True, **event_model_params
        )
        _, haz_1 = event_model(
            X=X, t=t, a=np.ones(X.shape[0]), return_p=True, **event_model_params
        )
        return haz_0, haz_1

    ts = range(BASE_T, t + BASE_T)
    if not risk:
        # compute survival probabilities
        hazards_t_0 = np.zeros([X.shape[0], len(ts)])
        hazards_t_1 = np.zeros([X.shape[0], len(ts)])
        hazards_c_0 = np.zeros([X.shape[0], len(ts)])
        hazards_c_1 = np.zeros([X.shape[0], len(ts)])
        idx = 0

        for t_hor in ts:
            _, hazards_t_0[:, idx] = event_model(
                X=X,
                t=t_hor,
                a=np.zeros(X.shape[0]),
                return_p=True,
                **event_model_params
            )
            _, hazards_t_1[:, idx] = event_model(
                X=X, t=t_hor, a=np.ones(X.shape[0]), return_p=True, **event_model_params
            )
            _, hazards_c_0[:, idx] = comp_model(
                X=X, t=t_hor, a=np.zeros(X.shape[0]), return_p=True, **comp_model_params
            )
            _, hazards_c_1[:, idx] = comp_model(
                X=X, t=t_hor, a=np.ones(X.shape[0]), return_p=True, **comp_model_params
            )
            idx = idx + 1

        # cumulate hazards into survival
        if surv_type == "cause-specific":
            surv_t_0 = np.cumprod((1.0 - hazards_t_0) * (1.0 - hazards_c_0), axis=1)[
                :, -1
            ]
            surv_t_1 = np.cumprod((1.0 - hazards_t_1) * (1.0 - hazards_c_1), axis=1)[
                :, -1
            ]
        elif surv_type == "risk-eliminated":
            surv_t_0 = np.cumprod((1.0 - hazards_t_0), axis=1)[:, -1]
            surv_t_1 = np.cumprod((1.0 - hazards_t_1), axis=1)[:, -1]
        elif surv_type == "separable":
            surv_t_0 = np.cumprod((1.0 - hazards_t_0) * (1.0 - hazards_c_0), axis=1)[
                :, -1
            ]
            surv_t_1 = np.cumprod((1.0 - hazards_t_1) * (1.0 - hazards_c_0), axis=1)[
                :, -1
            ]
        elif surv_type == "competing-risk-only":
            # assuming that this is the correct spec..
            surv_t_0 = np.cumprod((1.0 - hazards_c_0), axis=1)[:, -1]
            surv_t_1 = np.cumprod((1.0 - hazards_c_1), axis=1)[:, -1]
        else:
            raise ValueError("Unknown counterfactual survival type.")
    else:
        # compute risk of event occuring by time horizon
        surv_t_0, surv_t_1 = np.zeros((X.shape[0],)), np.zeros((X.shape[0],))
        for tsub in ts:
            tssub = range(BASE_T, tsub + BASE_T)
            hazards_t_0 = np.zeros([X.shape[0], len(tssub)])
            hazards_t_1 = np.zeros([X.shape[0], len(tssub)])
            hazards_c_0 = np.zeros([X.shape[0], len(tssub)])
            hazards_c_1 = np.zeros([X.shape[0], len(tssub)])
            idx = 0

            for t_hor in tssub:
                _, hazards_t_0[:, idx] = event_model(
                    X=X,
                    t=t_hor,
                    a=np.zeros(X.shape[0]),
                    return_p=True,
                    **event_model_params
                )
                _, hazards_t_1[:, idx] = event_model(
                    X=X,
                    t=t_hor,
                    a=np.ones(X.shape[0]),
                    return_p=True,
                    **event_model_params
                )
                _, hazards_c_0[:, idx] = comp_model(
                    X=X,
                    t=t_hor,
                    a=np.zeros(X.shape[0]),
                    return_p=True,
                    **comp_model_params
                )
                _, hazards_c_1[:, idx] = comp_model(
                    X=X,
                    t=t_hor,
                    a=np.ones(X.shape[0]),
                    return_p=True,
                    **comp_model_params
                )
                idx = idx + 1

            if tsub == BASE_T:
                if surv_type == "cause-specific":
                    surv_t_0 += (hazards_t_0 * (1.0 - hazards_c_0)).reshape(
                        -1,
                    )
                    surv_t_1 += (hazards_t_1 * (1.0 - hazards_c_1)).reshape(
                        -1,
                    )
                elif surv_type == "risk-eliminated":
                    surv_t_0 += hazards_t_0.reshape(
                        -1,
                    )
                    surv_t_1 += hazards_t_1.reshape(
                        -1,
                    )
                elif surv_type == "separable":
                    surv_t_0 += (hazards_t_0 * (1.0 - hazards_c_0)).reshape(
                        -1,
                    )
                    surv_t_1 += (hazards_t_1 * (1.0 - hazards_c_0)).reshape(
                        -1,
                    )
                else:
                    raise ValueError("Unknown counterfactual survival type.")
            else:
                if surv_type == "cause-specific":
                    surv_t_0 += (
                        hazards_t_0[:, -1]
                        * np.cumprod((1.0 - hazards_t_0) * (1.0 - hazards_c_0), axis=1)[
                            :, -2
                        ]
                    )
                    surv_t_1 += (
                        hazards_t_1[:, -1]
                        * np.cumprod((1.0 - hazards_t_1) * (1.0 - hazards_c_1), axis=1)[
                            :, -2
                        ]
                    )
                elif surv_type == "risk-eliminated":
                    surv_t_0 += (
                        hazards_t_0[:, -1]
                        * np.cumprod((1.0 - hazards_t_0), axis=1)[:, -2]
                    )
                    surv_t_1 += (
                        hazards_t_1[:, -1]
                        * np.cumprod((1.0 - hazards_t_1), axis=1)[:, -2]
                    )
                elif surv_type == "separable":
                    surv_t_0 += (
                        hazards_t_0[:, -1]
                        * np.cumprod((1.0 - hazards_t_0) * (1.0 - hazards_c_0), axis=1)[
                            :, -2
                        ]
                    )
                    surv_t_1 += (
                        hazards_t_1[:, -1]
                        * np.cumprod((1.0 - hazards_t_1) * (1.0 - hazards_c_0), axis=1)[
                            :, -2
                        ]
                    )
                else:
                    raise ValueError("Unknown counterfactual survival type.")

    return surv_t_0, surv_t_1


def rmst_from_hazard(
    X,
    t,
    event_model=None,
    comp_model=None,
    event_model_params=None,
    comp_model_params=None,
    surv_type="cause-specific",
):
    # compute rmst from models
    ts = range(BASE_T, t + BASE_T)
    surv_0, surv_1 = np.zeros([X.shape[0], len(ts)]), np.zeros([X.shape[0], len(ts)])

    for t_horizon in ts:
        surv_t_0, surv_t_1 = counterfactual_survival_prob_from_discrete_hazard_model(
            X,
            t_horizon,
            event_model,
            comp_model,
            event_model_params,
            comp_model_params,
            surv_type,
        )

        surv_0[:, (t_horizon - BASE_T)] = surv_t_0
        surv_1[:, (t_horizon - BASE_T)] = surv_t_1

    return np.sum(surv_0, axis=1), np.sum(surv_1, axis=1)


def weights_from_underlying_model(
    X, t, comp_model=None, comp_model_params=None, surv_type="separable", max_weight=50
):
    # compute ground truth importance weights from competing risks model
    if comp_model is None:
        comp_model = constant_event_model

    if comp_model_params is None:
        comp_model_params = {}

    ts = range(BASE_T, t + BASE_T)
    hazards_c_0 = np.zeros([X.shape[0], len(ts)])
    hazards_c_1 = np.zeros([X.shape[0], len(ts)])
    idx = 0

    for t in ts:
        _, hazards_c_0[:, idx] = comp_model(
            X=X, t=t, a=np.zeros(X.shape[0]), return_p=True, **comp_model_params
        )
        _, hazards_c_1[:, idx] = comp_model(
            X=X, t=t, a=np.ones(X.shape[0]), return_p=True, **comp_model_params
        )
        idx = idx + 1

    if surv_type == "risk-eliminated":
        weights_0 = np.cumprod(1 / (1.0 - hazards_c_0), axis=1)
        weights_1 = np.cumprod(1 / (1.0 - hazards_c_1), axis=1)
    elif surv_type == "separable":
        weights_0 = np.ones([X.shape[0], len(ts)])
        weights_1 = np.cumprod((1.0 - hazards_c_0) / (1.0 - hazards_c_1), axis=1)
    else:
        raise ValueError("Unknown counterfactual survival type.")

    if max_weight is not None:
        weights_0[weights_0 > max_weight] = max_weight
        weights_1[weights_1 > max_weight] = max_weight

    if len(weights_0.shape) == 1:
        weights_0 = weights_0.reshape((X.shape[0], 1))
        weights_1 = weights_1.reshape((X.shape[0], 1))
    return weights_0, weights_1


def _get_values_only(X):
    # wrapper to return only values of data frame
    if isinstance(X, pd.DataFrame):
        X = X.values
    return X
