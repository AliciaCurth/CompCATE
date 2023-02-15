import numpy as np
import pandas as pd

import os
import csv
import random

from compcate.dgp.data_utils import BASE_T, weights_from_underlying_model
from compcate.dgp.simulation_building_blocks import propensity_linear, constant_event_model, \
     propensity_constant, competing_model_linear
from compcate.model.model_competing_risk import TSepHazardModel, ConstantEstimator
from compcate.experiments.experiment_utils import rmse, download_if_needed

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import clone
from sklearn.preprocessing import MinMaxScaler, StandardScaler

RESULT_DIR = '../../results/twins/' if __name__ == '__main__' else 'results/twins/'
DATA_DIR = '../../data/' if __name__ == '__main__' else 'data/'

DATASET = "Twin_Data.csv.gz"
URL = "https://bitbucket.org/mvdschaar/mlforhealthlabpub/raw/0b0190bcd38a76c405c805f1ca774971fcd85233/data/twins/Twin_Data.csv.gz"


def _clean_twins_data(df):
    # Code excerpts taken from https://github.com/AliciaCurth/CATENets/blob/main/catenets/datasets
    # /dataset_twins.py
    # written by Bogdan Cebere

    cleaned_columns = []
    for col in df.columns:
        cleaned_columns.append(col.replace("'", "").replace("’", ""))
    df.columns = cleaned_columns

    # 8: factor not on certificate, 9: factor not classifiable --> np.nan --> mode imputation
    medrisk_list = [
        "anemia",
        "cardiac",
        "lung",
        "diabetes",
        "herpes",
        "hydra",
        "hemo",
        "chyper",
        "phyper",
        "eclamp",
        "incervix",
        "pre4000",
        "dtotord",
        "preterm",
        "renal",
        "rh",
        "uterine",
        "othermr",
    ]
    # 99: missing
    other_list = ["cigar", "drink", "wtgain", "gestat", "dmeduc", "nprevist"]

    other_list2 = ["pldel", "resstatb"]  # but no samples are missing..

    bin_list = ["dmar"] + medrisk_list
    con_list = ["dmage", "mpcb"] + other_list
    cat_list = ["adequacy"] + other_list2

    for feat in medrisk_list:
        df[feat] = df[feat].apply(lambda x: df[feat].mode()[0] if x in [8, 9] else x)

    for feat in other_list:
        df.loc[df[feat] == 99, feat] = df.loc[df[feat] != 99, feat].mean()

    df_features = df[con_list + bin_list]

    for feat in cat_list:
        df_features = pd.concat(
            [df_features, pd.get_dummies(df[feat], prefix=feat)], axis=1
        )

    # administrative censoring
    df.loc[df["outcome(t=0)"] == 9999, "outcome(t=0)"] = 365
    df.loc[df["outcome(t=1)"] == 9999, "outcome(t=1)"] = 365

    feat_list = [
        'dmage', 'mpcb', 'cigar', 'drink', 'wtgain', 'gestat', 'dmeduc', 'nprevist', 'dmar',
        'anemia', 'cardiac', 'lung',
        'diabetes', 'herpes', 'hydra', 'hemo', 'chyper', 'phyper', 'eclamp', 'incervix', 'pre4000',
        'dtotord', 'preterm',
        'renal', 'rh', 'uterine', 'othermr'
    ]

    x = np.asarray(df_features[feat_list])
    t0 = np.asarray(df[["outcome(t=0)"]]).reshape((-1,))
    t1 = np.asarray(df[["outcome(t=1)"]]).reshape((-1,))

    return x, t0, t1, feat_list


def import_twins(treatment_assn_model = None,
                 treatment_assn_params = None,
                 competing_model=None,
                 competing_model_params = None,
                 return_t0_only = False,
                 scl = 'minmax',
                 seed=1234,
                 tmax=30):
    """
    Parameters
    ----------
    filepath: path to the metabric dataset
    time-interval: size of dicretization interval
    treatment_type: categorical {'type1', 'type2'}
        'type1': Uniform
    censoring_type: categorical (default: None)
        'type1': Exponential
    Returns
    -------
        x: covariates
        y1, y0: potential event outcomes
        t1, t0: potential time-to-event/censoring
        a: treatment assignment
        t_max: max event time
        feat_list: list of feature names
    """
    np.random.seed(seed)
    random.seed(seed)

    download_if_needed(DATA_DIR + DATASET, http_url=URL)

    # load original data
    df = pd.read_csv(DATA_DIR + DATASET)

    x, t0, t1, feat_list = _clean_twins_data(df)

    if scl == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    if treatment_assn_model is None:
        treatment_assn_model = propensity_linear
    if treatment_assn_params is None:
        treatment_assn_params = {}

    a = treatment_assn_model(x, **treatment_assn_params)

    # consider only first tmax days
    t0[t0 > (tmax - 1)] = tmax
    t1[t1 > (tmax - 1)] = tmax

    # start counting at BASE_T (twins starts counting at 0)
    t0 = t0 + BASE_T
    t1 = t1 + BASE_T

    if return_t0_only:
        return x, t0, feat_list

    t_max = np.max([t1, t0]).astype(int)

    t = np.zeros_like(t0)

    t[a == 1] = t1[a == 1]
    t[a == 0] = t0[a == 0]

    # get competing risks times
    c0 = create_competing_risks_from_model(x, np.zeros_like(a), t_max, competing_model,
                                           competing_model_params)
    c1 = create_competing_risks_from_model(x, np.ones_like(a), t_max, competing_model,
                                           competing_model_params)
    c = a*c1 + (1-a)*c0

    out_t = np.fmin(t, c)
    delta = t < c

    return x, out_t, delta, a, t1, t0, c1, c0, t_max, feat_list


def run_twins_experiment(n_train=0.5,
                         comp_model=None,
                         comp_model_params=None,
                         treat_assn_model=None,
                         treat_assn_model_params=None,
                         n_exp=5,
                         effect_types=None,
                         model_names=None,
                         metrics = None,
                         vary_model='comp',
                         vary_param='xi_a',
                         vary_vals=[0, 0.5, 1, 2, 3],
                         save_file=False,
                         file_name='test',
                         get_support_covs_from_model=True,
                         model_c='lr-cv',
                         model_t='cons',
                         model_prop='lr-cv',
                         est_params_c=None,
                         est_params_t=None,
                         ts=[5, 10, 20, 31],
                         with_prop=True,
                         scl='minmax',
                         feat_slc_from_control_only: bool = False,
                         n_test=5700,
                         tmax=30
                         ):
    if effect_types is None:
        effect_types = ['cs', 'sep', 'elim']
    if model_names is None:
        model_names = ['nv', 'west', 'w', 'cf']
    if metrics is None:
        metrics = ['ess', 'rmst']

    # set parameters for semi-synthetic part of the experiments
    if comp_model_params is None:
        comp_model_params = {} if vary_model == 'comp' else {'xi': 0, 'xi_a': 0}
    if treat_assn_model is None:
        treat_assn_model = propensity_linear
    if comp_model is None:
        comp_model = constant_event_model

    if treat_assn_model_params is None:
        treat_assn_model_params = {}

    if est_params_t is None:
        est_params_t = {}
    if est_params_c is None:
        est_params_c = {}

    # choose variables for censoring and competing risk: they should be somewhat predictive of
    # outcome to ensure there is actual bias
    if feat_slc_from_control_only:
        # use only control group for this
        x, t0, feat_list = import_twins(return_t0_only=True)
        clf = RandomForestRegressor(n_estimators=30, random_state=42)
        clf.fit(x, t0.reshape(-1,))
    else:
        # use both groups for this
        x, out_t, delta, a, t1, t0, c1, c0, t_max, feat_list = import_twins(
            treatment_assn_model=propensity_constant, tmax=tmax)
        clf_outcome = ((a*t1 + (1-a)*t0) == 1).astype(int)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(x, clf_outcome)
    slct = SelectFromModel(clf, prefit=True)
    support_feats = [idx for idx, indicator in enumerate(slct.get_support()) if indicator]
    if not get_support_covs_from_model:
        support_feats = [idx for idx in np.random.choice(x.shape[1], len(support_feats),
                                                         replace=False)]

    # use the selected features to create events and propensity
    if vary_model == 'comp':
        comp_model_params.update({'support_covs': support_feats})

    treat_assn_model_params.update({'support_covs': support_feats})

    # setup estimators to use ------------------------------------------------------------
    if model_c == 'cons':
        model_params_c = {'po_estimator': ConstantEstimator}
    elif model_c == 'lr':
        model_params_c = {'po_estimator': LogisticRegression, 'solver': 'saga', 'C': 10e-3,
                          'penalty': 'l1', 'max_iter': 1000, **est_params_c}
    elif model_c == 'lr-cv':
        model_params_c = {'po_estimator': LogisticRegressionCV, 'solver': 'saga', 'cv': 3,
                          'penalty': 'l1', 'max_iter': 1000, 'Cs':[10e-1, 10e-3, 10e-2, 1],
                          **est_params_c}
    elif model_c == 'boost':
        model_params_c = {'po_estimator': GradientBoostingClassifier, **est_params_c}
    elif model_c == 'rf':
        model_params_c = {'po_estimator': RandomForestClassifier, 'n_estimators': 30,
                          **est_params_c}

    if model_t == 'cons':
        model_params_t = {'po_estimator': ConstantEstimator}
    elif model_t == 'lr':
        model_params_t = {'po_estimator': LogisticRegression, 'solver': 'saga', 'C': 10e-1,
                          'penalty': 'l1', 'max_iter': 1000, **est_params_t}
    elif model_t == 'lr-cv':
        model_params_t = {'po_estimator': LogisticRegressionCV, 'solver': 'saga', 'cv':3,
                          'penalty': 'l1', 'max_iter': 1000, 'Cs':[10e-3, 10e-2, 10e-1],
                          **est_params_t}
    elif model_t == 'boost':
        model_params_t = {'po_estimator': GradientBoostingClassifier, **est_params_t}
    elif model_t == 'rf':
        model_params_t = {'po_estimator': RandomForestClassifier, 'n_estimators': 100,
                          **est_params_t}

    if model_prop == 'lr':
        model_prop = LogisticRegression(C= 10e-3)
    elif model_prop == 'lr-cv':
        model_prop = LogisticRegressionCV(**{'solver': 'saga', 'cv':3,
                          'penalty': 'l1', 'max_iter': 1000, 'Cs':[10e-1, 10e-3, 10e-2]})

    # create dataframe to save all resuls in ------------------------------------------------
    header = ['seed', 'vary_model', 'vary_param', 'vary_value', 't_horizon']
    for setting in effect_types:
        for metric in metrics:
            header += [metric + '_0_' + setting + '_' + name for name in model_names]
            header += [metric + '_1_' + setting + '_' + name for name in model_names]
            if not metric == 'ess':
                header += [metric + '_te_' + setting + '_' + name for name in model_names]
            else:
                header += [metric + '_n0_' + setting + '_' + name for name in model_names]
                header += [metric + '_n1_' + setting + '_' + name for name in model_names]

    if save_file:
        # make path if it does not exist
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

        # get file to write in
        out_file = open(RESULT_DIR + (file_name + ".csv"), "w", buffering=1, newline='')
        writer = csv.writer(out_file)

        writer.writerow(header)

    result_frame = pd.DataFrame(columns=header)

    # create model set ---------------------------------------------------------------------
    models_to_fit = ['nv'] if 'nv' in model_names else []
    for setting in effect_types:
        models_to_fit += [setting + '_' + name for name in model_names if not (name == 'nv')]

    for seed in range(n_exp):  # loop over seeds
        for vary_val in vary_vals:  # loop over different experimental effect_types
            if vary_model == 'comp':
                comp_model_params.update({vary_param: vary_val})
            elif vary_model == 'treat':
                treat_assn_model_params.update({vary_param: vary_val})
            elif vary_model == 'n_train':
                n_train = vary_val
            else:
                raise ValueError("unknown vary_model")

            x, t, delta, a, t1, t0, c1, c0, t_max, feat_list = import_twins(
                treatment_assn_model=treat_assn_model,
                treatment_assn_params=treat_assn_model_params, competing_model=comp_model,
                competing_model_params=comp_model_params, seed=seed, scl=scl, tmax=tmax)

            X, x_t, t, t_t, delta, delta_t, a, a_t, \
            t1, t1_t, t0, t0_t, c1, c1_t, c0, c0_t = train_test_split(x, t, delta, a, t1, t0, c1,
                                                                      c0,
                                                                      train_size=n_train,
                                                                      random_state=seed)
            if n_test is not None:
                x_t = x_t[:n_test, :]
                t1_t = t1_t[:n_test]
                t0_t = t0_t[:n_test]
                c1_t = c1_t[:n_test]
                c0_t = c0_t[:n_test]

            model_dict = {}
            for name in models_to_fit:
                print('Experiment {}, vary_val {}: Fitting model {}.'.format(seed, vary_val, name))

                # create the different estimators
                if name == 'nv':
                    # naive estimator: just fit on observed data as is
                    nv = TSepHazardModel(model_t_params=model_params_t,
                                            model_c_params=model_params_c)
                    nv.fit(X, t, delta, a)
                    model_dict.update({'nv': nv})
                elif name == 'elim_cf':
                    # oracle estimator fit on counterfactual data
                    # create counterfactual data
                    a_elim = np.random.binomial(1, p=0.5*np.ones_like(a))  # remove confounding
                    t_elim = a_elim * t1 + (1-a_elim)*t0  # eliminate competing events
                    delta_elim = t_elim < 32

                    # fit on new data
                    elim_cf = TSepHazardModel(model_t_params=model_params_t,
                                             model_c_params=model_params_c,
                                             fit_comp=False
                                             )
                    elim_cf.fit(X, t_elim, delta_elim, a_elim)
                    model_dict.update({name: elim_cf})
                elif name == 'sep_cf':
                    # oracle estimator fit on counterfactual data
                    # create counterfactual data
                    a_sep = np.random.binomial(1, p=0.5*np.ones_like(a)) # remove confounding
                    t_raw_sep = a_sep * t1 + (1-a_sep)*t0
                    delta_sep = c0 > t_raw_sep # use c0 to separate out effect of a comp event
                    t_sep = np.fmin(t_raw_sep, c0)

                    # fit model on new data
                    sep_cf = TSepHazardModel(model_t_params=model_params_t,
                                                model_c_params=model_params_c)
                    sep_cf.fit(X, t_sep, delta_sep, a_sep)
                    model_dict.update({'sep_cf': sep_cf})

                elif name == 'cs_cf':
                    # oracle estimator fit on counterfactual data
                    # create counterfactual data
                    a_cs = np.random.binomial(1, p=0.5*np.ones_like(a)) # remove confounding only
                    t_raw_cs = a_cs * t1 + (1-a_cs) * t0
                    c_raw_cs = a_cs * c1 + (1-a_cs) * c0
                    delta_cs = c_raw_cs > t_raw_cs
                    t_cs = np.fmin(t_raw_cs, c_raw_cs)

                    cs_cf = TSepHazardModel(model_t_params=model_params_t,
                                           model_c_params=model_params_c)
                    cs_cf.fit(X, t_cs, delta_cs, a_cs)
                    model_dict.update({'cs_cf': cs_cf})

                elif name == 'elim_w':
                    # oracle estimator with ground truth weights for risk-eliminated effect
                    # compute ground truth weights for effect under elimination
                    weight_0_elim, weight_1_elim = weights_from_underlying_model(X, t_max,
                                                                                 comp_model=comp_model,
                                                                                 comp_model_params=comp_model_params,
                                                                                 surv_type='risk-eliminated')
                    if with_prop:
                        _, prop = treat_assn_model(X, return_p=True, **treat_assn_model_params)
                        weight_0_elim = weight_0_elim * (1 / (1 - prop))[:, None]
                        weight_1_elim = weight_1_elim * (1 / (prop))[:, None]

                    # fit model
                    elim_w = TSepHazardModel(model_t_params=model_params_t,
                                                model_c_params=model_params_c)
                    elim_w.fit(X, t, delta, a, sample_weight_treated=weight_1_elim,
                               sample_weight_control=weight_0_elim)
                    model_dict.update({'elim_w': elim_w})
                elif name == 'sep_w':
                    # oracle estimator with ground truth weights for separable effect
                    # compute ground truth weights for effect under separation
                    weight_0_sep, weight_1_sep = weights_from_underlying_model(X, t_max,
                                                                               comp_model=comp_model,
                                                                               comp_model_params=comp_model_params,
                                                                               surv_type='separable')
                    if with_prop:
                        _, prop = treat_assn_model(X, return_p=True, **treat_assn_model_params)
                        weight_0_sep = weight_0_sep * (1 / (1 - prop))[:, None]
                        weight_1_sep = weight_1_sep * (1 / (prop))[:, None]

                    # fit model
                    sep_w = TSepHazardModel(model_t_params=model_params_t,
                                               model_c_params=model_params_c)
                    sep_w.fit(X, t, delta, a, sample_weight_treated=weight_1_sep,
                              sample_weight_control=weight_0_sep)
                    model_dict.update({'sep_w': sep_w})
                elif name == 'cs_w':
                    # oracle estimator with ground truth weights for cause-specific effect
                    # compute ground truth weights eliminating only confounding
                    _, true_prop = treat_assn_model(X, return_p=True, **treat_assn_model_params)
                    true_prop = np.ones(t_max + 1) * true_prop[:, None]

                    cs_w = TSepHazardModel(model_t_params=model_params_t,
                                          model_c_params=model_params_c
                                          )
                    weight_0_cs = 1 / (1 - true_prop)
                    weight_1_cs = 1 / true_prop
                    cs_w.fit(X, t, delta, a,
                             sample_weight_control=weight_0_cs,
                             sample_weight_treated=weight_1_cs)
                    model_dict.update({'cs_w': cs_w})

                elif name == 'elim_west':
                    # estimator with estimated weights for risk-eliminated effect
                    elim_west = TSepHazardModel(model_t_params=model_params_t,
                                               model_c_params=model_params_c,
                                               est_weight='risk-eliminated',
                                               est_prop=with_prop,
                                               model_propensity=clone(model_prop))
                    elim_west.fit(X, t, delta, a)
                    model_dict.update({name: elim_west})

                elif name == 'sep_west':
                    # estimator with estimated weights for separable effect
                    sep_west = TSepHazardModel(model_t_params=model_params_t,
                                              model_c_params=model_params_c,
                                              est_weight='separable',
                                              est_prop=with_prop,
                                              model_propensity=clone(model_prop)
                                              )
                    sep_west.fit(X, t, delta, a)
                    model_dict.update({name: sep_west})
                elif name == 'cs_west':
                    # estimator with estimated weights for cause-specific effect
                    cs_west = TSepHazardModel(model_t_params=model_params_t,
                                             model_c_params=model_params_c,
                                             est_weight='cause-specific',
                                             est_prop=True,
                                             model_propensity=clone(model_prop)
                                             )
                    cs_west.fit(X, t, delta, a)
                    model_dict.update({name: cs_west})
                else:
                    raise ValueError("unknown model name {}".format(name))

            print(
                'Experiment {}, vary_val {}: Completed all model training.'.format(seed, vary_val))

            # evaluate metrics across all time horizons --------------------------------------
            for t_horizon in ts:
                res_list = []
                for setting in effect_types:
                    for metric in metrics:
                        # compute ground truth restricted to time-horizon
                        if setting == 'elim':
                            if metric == 'rmst':
                                true_0 = np.fmin(t0_t, t_horizon*np.ones_like(t0_t))
                                true_1 = np.fmin(t1_t, t_horizon * np.ones_like(t1_t))

                        elif setting == 'sep':
                            if metric == 'rmst':
                                t0_raw = np.fmin(t0_t, c0_t)
                                t1_raw = np.fmin(t1_t, c0_t)
                                true_0 = np.fmin(t0_raw, t_horizon*np.ones_like(t0_t))
                                true_1 = np.fmin(t1_raw, t_horizon * np.ones_like(t1_t))

                        elif setting == 'cs':
                            if metric == 'rmst':
                                t0_raw = np.fmin(t0_t, c0_t)
                                t1_raw = np.fmin(t1_t, c1_t)
                                true_0 = np.fmin(t0_raw, t_horizon * np.ones_like(t0_t))
                                true_1 = np.fmin(t1_raw, t_horizon * np.ones_like(t1_t))

                        res_0 = []
                        res_1 = []
                        res_te = []
                        res_ess_2 = []
                        for name in model_names:
                            key = setting + '_' + name if not (name == 'nv') else name
                            if metric == 'rmst':
                                pred_0, pred_1 = model_dict[key].predict_rmst(
                                x_t, t_horizon, surv_type='risk-eliminated' if
                                setting == 'elim' else ('separable' if setting == 'sep' else
                                                        'cause-specific'))
                            elif metric == 'ess':
                                # compute ess
                                if name == 'cf':
                                    ess_0, ess_1, n_0, n_1 = model_dict[
                                    key].compute_effective_sample_size(X,
                                                                       t_sep if setting ==
                                                                                   'sep' else
                                                                       (t_elim if setting
                                                                                     == 'elim' else t_cs),
                                                                       delta_sep if setting == 'sep' else (
                                                                           delta_elim if setting
                                                                                            == 'elim' else delta_cs),
                                                                       a_sep if setting ==
                                                                                   'sep' else (
                                                                           a_elim if setting
                                                                                        == 'elim' else a_cs),
                                                                       t_horizon)
                                elif name in ['nv', 'west']:
                                    ess_0, ess_1, n_0, n_1 = model_dict[
                                    key].compute_effective_sample_size(X, t, delta, a,
                                                                       t_horizon)
                                elif name == 'w':
                                    ess_0, ess_1, n_0, n_1 = model_dict[
                                    key].compute_effective_sample_size(X, t, delta, a,
                                                                       t_horizon,
                                                                       sample_weight_treated=weight_1_sep if setting == 'sep' else
                                                                       (
                                                                           weight_1_elim if setting == 'elim' else weight_1_cs),
                                                                       sample_weight_control=weight_0_sep if setting == 'sep' else
                                                                       (
                                                                           weight_0_elim if setting == 'elim' else weight_0_cs))

                            if metric == 'rmst':
                                res_0.append(rmse(true_0, pred_0))
                                res_1.append(rmse(true_1, pred_1))
                                res_te.append(rmse(true_1 - true_0, pred_1 - pred_0))
                            else:
                                res_0.append(ess_0)
                                res_1.append(ess_1)
                                res_te.append(n_0)
                                res_ess_2.append(n_1)

                        res_list = res_list + res_0 + res_1 + res_te
                        if metric == 'ess':
                            res_list += res_ess_2
                setting = [seed, vary_model, vary_param, vary_val, t_horizon]
                next_row = setting + res_list

                if save_file:
                    writer.writerow(next_row)

                new_frame = pd.DataFrame(columns=header, data=[next_row])
                result_frame = pd.concat([result_frame, new_frame])
    if save_file:
        out_file.close()

    return result_frame


def create_competing_risks_from_model(X, a, t_max, event_model, event_model_params=None):
    # function to simulate competing risks on top of true tte data
    if event_model_params is None:
        event_model_params = {}
    if event_model is None:
        event_model = constant_event_model

    comp_times = (t_max)*np.ones_like(a)
    idx_at_risk = np.ones_like(a)
    for t in range(BASE_T, t_max):
        # sample event indicators for everyone but only include for those at risk
        n_c_t = event_model(X, t, a, **event_model_params)
        comp_times[(idx_at_risk == 1) & (n_c_t == 1)] = t
        idx_at_risk = idx_at_risk & (n_c_t == 0)

    return comp_times


def run_experiment_by_setting(setting: str = '1', file_name='',
                              n_train=.5, n_exp=10, model_t='cons',
                              get_support_covs_from_model=True,
                              effect_types=None):
    if effect_types is None:
        effect_types = ['cs', 'sep', 'elim']

    if setting == '1':
        # no competing risk, just propensity
        run_twins_experiment(n_train=n_train,
                             save_file=True,
                             model_t=model_t,
                             ts=[2, 5, 10],
                             get_support_covs_from_model=True,
                             file_name=file_name + '_setting1',
                             treat_assn_model=propensity_linear,
                             vary_model='treat',
                             vary_param='xi',
                             tmax=10,
                             n_exp=n_exp,
                             effect_types=effect_types)
    elif setting == '2':
        # no propenisty, just competing risk
        run_twins_experiment(n_train=n_train,
                             save_file=True,
                             model_t=model_t,
                             ts=[2, 5, 10],
                             get_support_covs_from_model=get_support_covs_from_model,
                             file_name=file_name + '_setting2',
                             comp_model=competing_model_linear,
                             treat_assn_model=propensity_constant,
                             tmax=10, n_exp=n_exp,
                             effect_types=effect_types)
    elif setting == '3':
        # propensity and competing risk
        run_twins_experiment(n_train=n_train, save_file=True, model_t=model_t,
                             ts=[2, 5, 10], get_support_covs_from_model=True,
                             file_name=file_name + '_setting3',
                             comp_model=competing_model_linear,
                             treat_assn_model=propensity_linear,
                             tmax=10,
                             treat_assn_model_params={'xi': 2},
                             n_exp=n_exp,
                             effect_types=effect_types)
    else:
        raise ValueError('Twins setting should be 1, 2 or 3; you passed {}.'.format(setting))

if __name__ == '__main__':
    run_experiment_by_setting(setting='1')