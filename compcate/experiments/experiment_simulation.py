# Code to replicate the simulation experiments in the main paper
from compcate.dgp.data_utils import (
    counterfactual_survival_prob_from_discrete_hazard_model,
    weights_from_underlying_model,
    rmst_from_hazard,
    get_survivors_from_data,
)
from compcate.dgp.simulation_building_blocks import (
    constant_event_model,
    propensity_constant,
    propensity_linear,
    sim_discrete_time_survival_treatment_setup,
    binary_covariate_model,
    binary_hazard_model,
)
from compcate.model.model_competing_risk import TSepHazardModel, ConstantEstimator
from compcate.experiments.experiment_utils import rmse

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
import os
import csv

RESULT_DIR = "../../results/sim/" if __name__ == "__main__" else "results/sim/"


def run_experiment(
    ts=[1, 2, 5, 8, 10, 15, 20, 30],
    n_exp=10,
    n_train=200,
    n_test=10000,
    d=2,
    t_max=30,
    file_name="test",
    save_file=False,
    event_model=binary_hazard_model,
    comp_model=binary_hazard_model,
    covariate_model=binary_covariate_model,
    event_params=None,
    comp_params=None,
    covariate_model_params=None,
    treatment_assn_model=None,
    treatment_model_params=None,
    est_params_c=None,
    est_params_t=None,
    vary_model: str = "comp",
    vary_param="p_group_1",
    vary_values=[0.01, 0.05, 0.1, 0.2],
    effect_types=None,
    model_names=["nv", "cf", "w", "west"],
    metrics=["rmse", "rmst", "ess", "haz", "marghaz"],
    model_c="cons",
    model_t="cons",
    with_prop: bool = False,
    risk=True,
):
    if effect_types is None:
        effect_types = ["elim", "sep", "cs"]
    # create frame to save results in------------------------------------------
    header = ["seed", "vary_model", "vary_param", "vary_value", "t_horizon"]
    for setting in effect_types:
        for metric in metrics:
            header += [metric + "_0_" + setting + "_" + name for name in model_names]
            header += [metric + "_1_" + setting + "_" + name for name in model_names]
            if not metric == "ess":
                header += [
                    metric + "_te_" + setting + "_" + name for name in model_names
                ]
            else:
                header += [
                    metric + "_n0_" + setting + "_" + name for name in model_names
                ]
                header += [
                    metric + "_n1_" + setting + "_" + name for name in model_names
                ]

    if save_file:
        # make path if it does not exist
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)

        # get file to write in
        out_file = open(RESULT_DIR + (file_name + ".csv"), "w", buffering=1, newline="")
        writer = csv.writer(out_file)

        writer.writerow(header)

    result_frame = pd.DataFrame(columns=header)

    # set parameters for data generating process --------------------------------------------
    if event_params is None:
        event_params = {}
    if comp_params is None:
        comp_params = {}
    if treatment_model_params is None:
        treatment_model_params = {}

    if treatment_assn_model is None:
        treatment_assn_model = propensity_constant

    if covariate_model_params is None:
        covariate_model_params = {}

    if est_params_c is None:
        est_params_c = {}
    if est_params_t is None:
        est_params_t = {}

    # define parameters for the used estimators ---------------------------------------------
    if model_c == "cons":
        model_params_c = {"po_estimator": ConstantEstimator}
    elif model_c == "lr":
        model_params_c = {"po_estimator": LogisticRegression, **est_params_c}
    elif model_c == "boost":
        model_params_c = {"po_estimator": GradientBoostingClassifier, **est_params_c}

    if model_t == "cons":
        model_params_t = {"po_estimator": ConstantEstimator}
    elif model_t == "lr":
        model_params_t = {"po_estimator": LogisticRegression, **est_params_t}
    elif model_t == "boost":
        model_params_t = {"po_estimator": GradientBoostingClassifier, **est_params_t}

    # define the set of models to fit ------------------------------------------------------------
    models_to_fit = ["nv"] if "nv" in model_names else []
    for setting in effect_types:
        models_to_fit += [
            setting + "_" + name for name in model_names if not (name == "nv")
        ]

    # loop over all seeds + 'vary_values' in the experiment --------------------------------------
    for seed in range(n_exp):
        for vary_val in vary_values:
            # pick the right value to loop over ----------
            if vary_model == "comp":
                comp_params.update({vary_param: vary_val})
            elif vary_model == "event":
                event_params.update({vary_param: vary_val})
            elif vary_model == "cov":
                covariate_model_params.update({vary_param: vary_val})
            elif vary_model == "treat":
                treatment_model_params.update({vary_param: vary_val})
            elif vary_model == "n_train":
                n_train = vary_val
            else:
                raise ValueError("unknown vary_model")

            # generate data based on model ----------------------------------------
            X, a, t, delta = sim_discrete_time_survival_treatment_setup(
                n=n_train + n_test,
                d=d,
                t_max=t_max,
                event_model=event_model,
                competing_model=comp_model,
                return_short=True,
                return_long=False,
                event_model_params=event_params,
                competing_model_params=comp_params,
                seed=seed,
                covariate_model=covariate_model,
                covariate_model_params=covariate_model_params,
                treatment_assn_model=treatment_assn_model,
                treatment_assn_model_params=treatment_model_params,
            )
            X_test = X[n_train : (n_train + n_test), :].copy()
            X, a, t, delta = X[:n_train, :], a[:n_train], t[:n_train], delta[:n_train]

            # fit models: loop over all models and fit in turn --------------------------
            model_dict = {}
            for name in models_to_fit:
                print(
                    "Experiment {}, vary_val {}: Fitting model {}.".format(
                        seed, vary_val, name
                    )
                )
                if name == "nv":
                    # naive estimator: just fit a standard model
                    nv = TSepHazardModel(
                        model_t_params=model_params_t, model_c_params=model_params_c
                    )
                    nv.fit(X, t, delta, a)
                    model_dict.update({"nv": nv})
                elif name == "elim_cf":
                    # (oracle) model fit on counterfactual data in which the competing event was
                    # eliminated

                    # need new data to do this
                    (
                        X_cf_elim,
                        a_cf_elim,
                        t_cf_elim,
                        delta_cf_elim,
                    ) = sim_discrete_time_survival_treatment_setup(
                        n=n_train,
                        d=d,
                        t_max=t_max,
                        event_model=event_model,
                        competing_model=constant_event_model,
                        return_short=True,
                        return_long=False,
                        event_model_params=event_params,
                        competing_model_params={
                            # no competing event at all
                            "xi": 0,
                            "xi_a": 0,
                        },
                        seed=41 * (seed + 1),
                        X_given=X,
                    )

                    # fit new model on oracle data
                    elim_cf = TSepHazardModel(
                        fit_comp=False,
                        model_t_params=model_params_t,
                        model_c_params=model_params_c,
                    )
                    elim_cf.fit(
                        X_cf_elim[:n_train, :],
                        t_cf_elim[:n_train],
                        delta_cf_elim[:n_train],
                        a_cf_elim[:n_train],
                    )
                    model_dict.update({"elim_cf": elim_cf})

                    if "haz" in metrics:
                        # also generate counterfactual test data to have a population to evaluate
                        # hazard against later
                        (
                            X_cf_elim_test,
                            a_cf_elim_test,
                            t_cf_elim_test,
                            delta_cf_elim_test,
                        ) = sim_discrete_time_survival_treatment_setup(
                            n=n_test,
                            d=d,
                            t_max=t_max,
                            event_model=event_model,
                            competing_model=constant_event_model,
                            return_short=True,
                            return_long=False,
                            event_model_params=event_params,
                            competing_model_params={
                                # no competing at all
                                "xi": 0,
                                "xi_a": 0,
                            },
                            seed=41 * (seed + 1),
                            X_given=X_test,
                        )
                elif name == "sep_cf":
                    # (oracle) model fit on counterfactual data in which the treatment effect on
                    # the competing event was eliminated

                    # regenerate data
                    comp_params_sep = comp_params.copy()
                    comp_params_sep.update({"counterfactual_sep": True})
                    (
                        X_cf_sep,
                        a_cf_sep,
                        t_cf_sep,
                        delta_cf_sep,
                    ) = sim_discrete_time_survival_treatment_setup(
                        n=n_train,
                        d=d,
                        t_max=t_max,
                        event_model=event_model,
                        competing_model=comp_model,
                        return_short=True,
                        return_long=False,
                        event_model_params=event_params,
                        competing_model_params=comp_params_sep,
                        seed=41 * (seed + 1),
                        X_given=X,
                    )

                    # fit model on new data
                    sep_cf = TSepHazardModel(
                        model_t_params=model_params_t, model_c_params=model_params_c
                    )
                    sep_cf.fit(
                        X_cf_sep[:n_train, :],
                        t_cf_sep[:n_train],
                        delta_cf_sep[:n_train],
                        a_cf_sep[:n_train],
                    )
                    model_dict.update({"sep_cf": sep_cf})

                    if "haz" in metrics:
                        # also generate counterfactual test data to have a population to evaluate
                        # hazard against later
                        (
                            X_cf_sep_test,
                            a_cf_sep_test,
                            t_cf_sep_test,
                            delta_cf_sep_test,
                        ) = sim_discrete_time_survival_treatment_setup(
                            n=n_test,
                            d=d,
                            t_max=t_max,
                            event_model=event_model,
                            competing_model=comp_model,
                            return_short=True,
                            return_long=False,
                            event_model_params=event_params,
                            competing_model_params=comp_params_sep,
                            seed=42 * (seed + 1),
                            X_given=X_test,
                        )
                elif name == "cs_cf":
                    # (oracle) model fit on counterfactual data in which there is no treatment
                    # assignment bias; all other shifts remain (cause-specific TE of interest)
                    (
                        X_cf_cs,
                        a_cf_cs,
                        t_cf_cs,
                        delta_cf_cs,
                    ) = sim_discrete_time_survival_treatment_setup(
                        n=n_train,
                        d=d,
                        t_max=t_max,
                        event_model=event_model,
                        competing_model=comp_model,
                        return_short=True,
                        return_long=False,
                        event_model_params=event_params,
                        competing_model_params=comp_params,
                        seed=43 * (seed + 1),
                        covariate_model=covariate_model,
                        covariate_model_params=covariate_model_params,
                        X_given=X,
                    )

                    # fit model on new data
                    cs_cf = TSepHazardModel(
                        model_t_params=model_params_t, model_c_params=model_params_c
                    )
                    cs_cf.fit(
                        X_cf_cs[:n_train, :],
                        t_cf_cs[:n_train],
                        delta_cf_cs[:n_train],
                        a_cf_cs[:n_train],
                    )
                    model_dict.update({"cs_cf": cs_cf})
                    if "haz" in metrics:
                        # also generate counterfactual test data to have a population to evaluate
                        # hazard against later
                        (
                            X_cf_cs_test,
                            a_cf_cs_test,
                            t_cf_cs_test,
                            delta_cf_cs_test,
                        ) = sim_discrete_time_survival_treatment_setup(
                            n=n_test,
                            d=d,
                            t_max=t_max,
                            event_model=event_model,
                            competing_model=comp_model,
                            return_short=True,
                            return_long=False,
                            event_model_params=event_params,
                            competing_model_params=comp_params,
                            seed=43 * (seed + 1),
                            covariate_model=covariate_model,
                            covariate_model_params=covariate_model_params,
                            X_given=X_test,
                        )

                elif name == "elim_w":
                    # oracle estimator with ground truth weights for risk-eliminated effect
                    # compute ground truth weights for effect under elimination
                    weight_0_elim, weight_1_elim = weights_from_underlying_model(
                        X,
                        t_max,
                        comp_model=comp_model,
                        comp_model_params=comp_params,
                        surv_type="risk-eliminated",
                    )
                    if with_prop:
                        # if we are also adjusting for propensity
                        _, prop = treatment_assn_model(
                            X, return_p=True, **treatment_model_params
                        )
                        weight_0_elim = weight_0_elim * (1 / (1 - prop))[:, None]
                        weight_1_elim = weight_1_elim * (1 / (prop))[:, None]

                    # fit weighted model
                    elim_w = TSepHazardModel(
                        model_t_params=model_params_t, model_c_params=model_params_c
                    )
                    elim_w.fit(
                        X,
                        t,
                        delta,
                        a,
                        sample_weight_treated=weight_1_elim,
                        sample_weight_control=weight_0_elim,
                    )

                    model_dict.update({"elim_w": elim_w})
                elif name == "sep_w":
                    # oracle estimator with ground truth weights for separable effect
                    # compute ground truth weights for separable effect
                    weight_0_sep, weight_1_sep = weights_from_underlying_model(
                        X,
                        t_max,
                        comp_model=comp_model,
                        comp_model_params=comp_params,
                        surv_type="separable",
                    )

                    if with_prop:
                        # if we are also adjusting for propensity
                        _, prop = treatment_assn_model(
                            X, return_p=True, **treatment_model_params
                        )
                        weight_0_sep = weight_0_sep * (1 / (1 - prop))[:, None]
                        weight_1_sep = weight_1_sep * (1 / (prop))[:, None]

                    sep_w = TSepHazardModel(
                        model_t_params=model_params_t, model_c_params=model_params_c
                    )
                    sep_w.fit(
                        X,
                        t,
                        delta,
                        a,
                        sample_weight_treated=weight_1_sep,
                        sample_weight_control=weight_0_sep,
                    )
                    model_dict.update({"sep_w": sep_w})

                elif name == "cs_w":
                    # oracle estimator with ground truth weights for cause-specific effect
                    # compute ground truth weights for cause specific effects (propensity only)
                    _, true_prop = treatment_assn_model(
                        X, return_p=True, **treatment_model_params
                    )
                    true_prop = np.ones(t_max + 1) * true_prop[:, None]

                    cs_w = TSepHazardModel(
                        model_t_params=model_params_t, model_c_params=model_params_c
                    )
                    weight_0_cs = 1 / (1 - true_prop)
                    weight_1_cs = 1 / true_prop
                    cs_w.fit(
                        X,
                        t,
                        delta,
                        a,
                        sample_weight_control=weight_0_cs,
                        sample_weight_treated=weight_1_cs,
                    )
                    model_dict.update({"cs_w": cs_w})

                elif name == "elim_west":
                    # estimator with estimated weights for risk-eliminated effect
                    elim_west = TSepHazardModel(
                        model_t_params=model_params_t,
                        model_c_params=model_params_c,
                        est_weight="risk-eliminated",
                        est_prop=with_prop,
                    )
                    elim_west.fit(X, t, delta, a)
                    model_dict.update({name: elim_west})

                elif name == "sep_west":
                    # estimator with estimated weights for separable effect
                    sep_west = TSepHazardModel(
                        model_t_params=model_params_t,
                        model_c_params=model_params_c,
                        est_weight="separable",
                        est_prop=with_prop,
                    )
                    sep_west.fit(X, t, delta, a)
                    model_dict.update({name: sep_west})
                elif name == "cs_west":
                    # estimator with estimated weights for cause-specific effect
                    cs_west = TSepHazardModel(
                        model_t_params=model_params_t,
                        model_c_params=model_params_c,
                        est_weight="cause-specific",
                        est_prop=True,
                    )
                    cs_west.fit(X, t, delta, a)
                    model_dict.update({name: cs_west})
                else:
                    raise ValueError("unknown model name {}".format(name))

            print(
                "Experiment {}, vary_value {}: Completed all model training.".format(
                    seed, vary_val
                )
            )

            # evaluate all fitted models  -------------------------------------------------------
            for t_horizon in ts:  # across different time horizons
                res_list = []
                for (
                    setting
                ) in effect_types:  # across different effect_types (elim, sep, cs)
                    for metric in metrics:  # evaluate all metrics
                        if setting == "elim":  # risk-eliminated effects
                            if metric == "rmse":
                                # compute ground truth
                                (
                                    true_0,
                                    true_1,
                                ) = counterfactual_survival_prob_from_discrete_hazard_model(
                                    X_test,
                                    t_horizon,
                                    surv_type="risk-eliminated",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                    risk=risk,
                                )
                            elif metric == "rmst":
                                # compute ground truth
                                true_0, true_1 = rmst_from_hazard(
                                    X_test,
                                    t_horizon,
                                    surv_type="risk-eliminated",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                )
                            elif metric == "haz":
                                # get interventional population to evaluate against
                                X_cf_0_test, X_cf_1_test = get_survivors_from_data(
                                    X_cf_elim_test,
                                    t_cf_elim_test,
                                    delta_cf_elim_test,
                                    a_cf_elim_test,
                                    t_horizon=t_horizon,
                                )
                                # compute ground truth
                                (
                                    true_0,
                                    _,
                                ) = counterfactual_survival_prob_from_discrete_hazard_model(
                                    X_cf_0_test,
                                    t_horizon,
                                    surv_type="risk-eliminated",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                    haz_only=True,
                                )
                                (
                                    _,
                                    true_1,
                                ) = counterfactual_survival_prob_from_discrete_hazard_model(
                                    X_cf_1_test,
                                    t_horizon,
                                    surv_type="risk-eliminated",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                    haz_only=True,
                                )
                            elif metric == "marghaz":
                                # compute hazard on the observed population (not interventional)
                                (
                                    true_0,
                                    true_1,
                                ) = counterfactual_survival_prob_from_discrete_hazard_model(
                                    X_test,
                                    t_horizon,
                                    surv_type="risk-eliminated",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                    haz_only=True,
                                )
                        elif setting == "sep":  # separable effects
                            if metric == "rmse":
                                # compute ground truth
                                (
                                    true_0,
                                    true_1,
                                ) = counterfactual_survival_prob_from_discrete_hazard_model(
                                    X_test,
                                    t_horizon,
                                    surv_type="separable",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                    risk=risk,
                                )
                            elif metric == "rmst":
                                # compute ground truth
                                true_0, true_1 = rmst_from_hazard(
                                    X_test,
                                    t_horizon,
                                    surv_type="separable",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                )
                            elif metric == "haz":
                                # get interventional population
                                X_cf_0_test, X_cf_1_test = get_survivors_from_data(
                                    X_cf_sep_test,
                                    t_cf_sep_test,
                                    delta_cf_sep_test,
                                    a_cf_sep_test,
                                    t_horizon=t_horizon,
                                )
                                # compute ground truth
                                (
                                    true_0,
                                    _,
                                ) = counterfactual_survival_prob_from_discrete_hazard_model(
                                    X_cf_0_test,
                                    t_horizon,
                                    surv_type="separable",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                    haz_only=True,
                                )

                                (
                                    _,
                                    true_1,
                                ) = counterfactual_survival_prob_from_discrete_hazard_model(
                                    X_cf_1_test,
                                    t_horizon,
                                    surv_type="separable",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                    haz_only=True,
                                )
                            elif metric == "marghaz":
                                # compute ground truth on observed (not interventional) population
                                (
                                    true_0,
                                    true_1,
                                ) = counterfactual_survival_prob_from_discrete_hazard_model(
                                    X_test,
                                    t_horizon,
                                    surv_type="separable",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                    haz_only=True,
                                )
                        elif setting == "cs":  # cause-specific effects
                            if metric == "rmse":
                                # compute ground truth
                                (
                                    true_0,
                                    true_1,
                                ) = counterfactual_survival_prob_from_discrete_hazard_model(
                                    X_test,
                                    t_horizon,
                                    surv_type="cause-specific",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                    risk=risk,
                                )
                            elif metric == "rmst":
                                # compute ground truth
                                true_0, true_1 = rmst_from_hazard(
                                    X_test,
                                    t_horizon,
                                    surv_type="cause-specific",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                )
                            elif metric == "haz":
                                # generate counterfactual at risk population
                                X_cf_0_test, X_cf_1_test = get_survivors_from_data(
                                    X_cf_cs_test,
                                    t_cf_cs_test,
                                    delta_cf_cs_test,
                                    a_cf_cs_test,
                                    t_horizon=t_horizon,
                                )
                                # compute ground truth
                                (
                                    true_0,
                                    _,
                                ) = counterfactual_survival_prob_from_discrete_hazard_model(
                                    X_cf_0_test,
                                    t_horizon,
                                    surv_type="cause-specific",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                    haz_only=True,
                                )
                                (
                                    _,
                                    true_1,
                                ) = counterfactual_survival_prob_from_discrete_hazard_model(
                                    X_cf_1_test,
                                    t_horizon,
                                    surv_type="cause-specific",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                    haz_only=True,
                                )
                            elif metric == "marghaz":
                                # compute ground truth on observational (not interventional)
                                # at-risk population
                                (
                                    true_0,
                                    true_1,
                                ) = counterfactual_survival_prob_from_discrete_hazard_model(
                                    X_test,
                                    t_horizon,
                                    surv_type="cause-specific",
                                    event_model=event_model,
                                    comp_model=comp_model,
                                    event_model_params=event_params,
                                    comp_model_params=comp_params,
                                    haz_only=True,
                                )

                        res_0 = []
                        res_1 = []
                        res_te = []
                        res_ess_2 = []
                        # compute metrics: predict and compute --------------------------------
                        for name in model_names:
                            key = setting + "_" + name if not (name == "nv") else name
                            # compute predictions by model ----------------------------------------
                            if metric == "rmst":
                                pred_0, pred_1 = model_dict[key].predict_rmst(
                                    X_test,
                                    t_horizon,
                                    surv_type="risk-eliminated"
                                    if setting == "elim"
                                    else (
                                        "separable"
                                        if setting == "sep"
                                        else "cause-specific"
                                    ),
                                )
                            elif metric == "rmse":
                                if setting == "elim":
                                    pred_0, pred_1 = model_dict[
                                        key
                                    ].predict_risk_eliminated_survival(
                                        X_test, t_horizon, risk=risk
                                    )
                                elif setting == "sep":
                                    pred_0, pred_1 = model_dict[
                                        key
                                    ].predict_separable_survival(
                                        X_test, t_horizon, risk=risk
                                    )
                                elif setting == "cs":
                                    pred_0, pred_1 = model_dict[
                                        key
                                    ].predict_cause_specific_survivals(
                                        X_test, t_horizon, risk=risk
                                    )
                            elif metric == "haz":
                                # correct test population
                                pred_0, _ = model_dict[key].predict_hazard(
                                    X_cf_0_test, t_horizon, return_comp=False
                                )
                                _, pred_1 = model_dict[key].predict_hazard(
                                    X_cf_1_test, t_horizon, return_comp=False
                                )
                            elif metric == "marghaz":
                                pred_0, pred_1 = model_dict[key].predict_hazard(
                                    X_test, t_horizon, return_comp=False
                                )
                            elif metric == "ess":
                                # compute effective sample sizes
                                if name == "cf":
                                    ess_0, ess_1, n_0, n_1 = model_dict[
                                        key
                                    ].compute_effective_sample_size(
                                        X,
                                        t_cf_sep
                                        if setting == "sep"
                                        else (
                                            t_cf_elim if setting == "elim" else t_cf_cs
                                        ),
                                        delta_cf_sep
                                        if setting == "sep"
                                        else (
                                            delta_cf_elim
                                            if setting == "elim"
                                            else delta_cf_cs
                                        ),
                                        a_cf_sep
                                        if setting == "sep"
                                        else (
                                            a_cf_elim if setting == "elim" else a_cf_cs
                                        ),
                                        t_horizon,
                                    )
                                elif name in ["nv", "west"]:
                                    ess_0, ess_1, n_0, n_1 = model_dict[
                                        key
                                    ].compute_effective_sample_size(
                                        X, t, delta, a, t_horizon
                                    )
                                elif name == "w":
                                    ess_0, ess_1, n_0, n_1 = model_dict[
                                        key
                                    ].compute_effective_sample_size(
                                        X,
                                        t,
                                        delta,
                                        a,
                                        t_horizon,
                                        sample_weight_treated=weight_1_sep
                                        if setting == "sep"
                                        else (
                                            weight_1_elim
                                            if setting == "elim"
                                            else weight_1_cs
                                        ),
                                        sample_weight_control=weight_0_sep
                                        if setting == "sep"
                                        else (
                                            weight_0_elim
                                            if setting == "elim"
                                            else weight_0_cs
                                        ),
                                    )

                            # compute metrics from predictions and ground truth ----------------
                            if not metric == "ess":
                                res_0.append(rmse(true_0, pred_0))
                                res_1.append(rmse(true_1, pred_1))
                                if (metric == "haz") or (metric == "marghaz"):
                                    res_te.append(np.nan)
                                else:
                                    res_te.append(
                                        rmse(true_1 - true_0, pred_1 - pred_0)
                                    )
                            else:
                                res_0.append(ess_0)
                                res_1.append(ess_1)
                                res_te.append(n_0)
                                res_ess_2.append(n_1)

                        res_list = res_list + res_0 + res_1 + res_te
                        if metric == "ess":
                            res_list += res_ess_2

                # concatenate to results --------------------------------------------------------
                setting = [seed, vary_model, vary_param, vary_val, t_horizon]
                next_row = setting + res_list

                if save_file:
                    writer.writerow(next_row)

                new_frame = pd.DataFrame(columns=header, data=[next_row])
                result_frame = pd.concat([result_frame, new_frame])

    if save_file:
        out_file.close()

    return result_frame


def run_experiment_by_setting(
    setting=2,
    file_name="",
    n_train=5000,
    n_exp=10,
    return_res: bool = False,
    effect_types=None,
):
    if effect_types is None:
        effect_types = ["cs", "sep", "elim"]

    if setting == "1":
        # treatment has no effect, there is only confounding
        res = run_experiment(
            n_train=n_train,
            event_params={
                "p_group_0": 0.01,
                "p_group_1": 0.1,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            comp_params={
                "p_group_0": 0.01,
                "p_group_1": 0.01,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            vary_model="treat",
            vary_param="xi",
            t_max=30,
            vary_values=[0, 2, 4, 6, -2, -4, -6],
            ts=[1, 2, 5, 8, 10, 15, 20, 30],
            model_c="lr",
            model_t="cons",
            covariate_model_params={"cov_p": 0.35},
            est_params_c={"C": 100},
            effect_types=effect_types,
            model_names=["nv", "cf", "w", "west"],
            with_prop=True,
            treatment_assn_model=propensity_linear,
            treatment_model_params={"support_covs": [0]},
            file_name=file_name + "_setting1",
            save_file=True,
            n_exp=n_exp,
        )

    elif setting == "2":
        # treatment has heterogeneous effect only on competing risk
        res = run_experiment(
            n_train=n_train,
            event_params={
                "p_group_0": 0.01,
                "p_group_1": 0.1,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            comp_params={
                "p_group_0": 0.01,
                "p_group_1": 0.01,
                "p_a_group_0": 0,
                "support_cov": 0,
            },
            vary_param="p_a_group_1",
            t_max=30,
            vary_values=[0, 0.01, 0.1, 0.2],
            ts=[1, 2, 5, 8, 10, 15, 20, 30],
            model_c="lr",
            model_t="cons",
            covariate_model_params={"cov_p": 0.35},
            est_params_c={"C": 100},
            effect_types=effect_types,
            model_names=["nv", "cf", "w", "west"],
            save_file=True,
            file_name=file_name + "_setting2",
            n_exp=n_exp,
            risk=True,
        )
    elif setting == "3":
        # treaetment has effect only on main event
        res = run_experiment(
            n_train=n_train,
            event_params={
                "p_group_0": 0.01,
                "p_group_1": 0.1,
                "p_a_group_0": 0,
                "p_a_group_1": -0.09,
                "support_cov": 0,
            },
            comp_params={
                "p_group_0": 0.01,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            vary_param="p_group_1",
            t_max=30,
            vary_values=[0.01, 0.1, 0.2],
            ts=[1, 2, 5, 8, 10, 15, 20, 30],
            model_c="lr",
            model_t="cons",
            covariate_model_params={"cov_p": 0.35},
            est_params_c={"C": 100},
            effect_types=effect_types,
            model_names=["nv", "cf", "w", "west"],
            save_file=True,
            file_name=file_name + "_setting3",
            n_exp=n_exp,
            risk=True,
        )
    elif setting == "4":
        res = run_experiment(
            n_train=n_train,
            event_params={
                "p_group_0": 0.01,
                "p_group_1": 0.1,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            comp_params={
                "p_group_0": 0.01,
                "p_group_1": 0.01,
                "p_a_group_0": 0,
                "p_a_group_1": 0.1,
                "support_cov": 0,
            },
            vary_model="treat",
            vary_param="xi",
            t_max=30,
            vary_values=[0, 2, 4, 6, -2, -4, -6],
            ts=[1, 2, 5, 8, 10, 15, 20, 30],
            model_c="lr",
            model_t="lr",
            covariate_model_params={"cov_p": 0.35},
            est_params_c={"C": 100},
            effect_types=effect_types,
            model_names=["nv", "cf", "w", "west"],
            with_prop=True,
            treatment_assn_model=propensity_linear,
            treatment_model_params={"support_covs": [0]},
            save_file=True,
            file_name=file_name + "_setting4",
            n_exp=n_exp,
            risk=True,
        )
    elif setting == "1A":
        # no overlap between x_A and x_E/x_D and no correlation (setting same as setting 1
        # otherwise)
        res = run_experiment(
            n_train=n_train,
            event_params={
                "p_group_0": 0.01,
                "p_group_1": 0.1,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            comp_params={
                "p_group_0": 0.01,
                "p_group_1": 0.01,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            vary_model="treat",
            vary_param="xi",
            t_max=30,
            vary_values=[0, 2, 4, 6, -2, -4, -6],
            ts=[1, 2, 5, 8, 10, 15, 20, 30],
            model_c="lr",
            model_t="cons",
            covariate_model_params={"cov_p": 0},
            est_params_c={"C": 100},
            effect_types=effect_types,
            model_names=["nv", "cf", "w", "west"],
            with_prop=True,
            treatment_assn_model=propensity_linear,
            treatment_model_params={"support_covs": [1]},
            file_name=file_name + "_setting1A",
            n_exp=n_exp,
            risk=True,
            save_file=True,
        )

    elif setting == "1B":
        # no overlap between x_A and x_E/x_D but correlation between covs (setting same as
        # setting 1 otherwise)
        res = run_experiment(
            n_train=5000,
            event_params={
                "p_group_0": 0.01,
                "p_group_1": 0.1,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            comp_params={
                "p_group_0": 0.01,
                "p_group_1": 0.01,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            vary_model="treat",
            vary_param="xi",
            t_max=30,
            vary_values=[0, 2, 4, 6, -2, -4, -6],
            ts=[1, 2, 5, 8, 10, 15, 20, 30],
            model_c="lr",
            model_t="cons",
            covariate_model_params={"cov_p": 0.35},
            est_params_c={"C": 100},
            effect_types=effect_types,
            model_names=["nv", "cf", "w", "west"],
            with_prop=True,
            treatment_assn_model=propensity_linear,
            treatment_model_params={"support_covs": [1]},
            file_name=file_name + "_setting1B",
            n_exp=n_exp,
            risk=True,
            save_file=True,
        )
    elif setting == "1C":
        # correctly specified logistic regression (setting same as setting 1 otherwise)
        res = run_experiment(
            n_train=5000,
            event_params={
                "p_group_0": 0.01,
                "p_group_1": 0.1,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            comp_params={
                "p_group_0": 0.01,
                "p_group_1": 0.01,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            vary_model="treat",
            vary_param="xi",
            t_max=30,
            vary_values=[0, 2, 4, 6, -2, -4, -6],
            ts=[1, 2, 5, 8, 10, 15, 20, 30],
            model_c="lr",
            model_t="lr",
            covariate_model_params={"cov_p": 0.35},
            est_params_c={"C": 100},
            est_params_t={"C": 100},
            effect_types=effect_types,
            model_names=["nv", "cf", "w", "west"],
            with_prop=True,
            treatment_assn_model=propensity_linear,
            treatment_model_params={"support_covs": [0]},
            file_name=file_name + "_setting1C",
            n_exp=n_exp,
            risk=True,
            save_file=True,
        )

    elif setting == "1D":
        # misspecified logistic regression (setting same as 1 otherwsie)
        res = run_experiment(
            n_train=5000,
            event_params={
                "p_group_0": 0.01,
                "p_group_1": 0.1,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            comp_params={
                "p_group_0": 0.01,
                "p_group_1": 0.01,
                "p_a_group_0": 0,
                "p_a_group_1": 0,
                "support_cov": 0,
            },
            vary_model="treat",
            vary_param="xi",
            t_max=30,
            vary_values=[0, 2, 4, 6, -2, -4, -6],
            ts=[1, 2, 5, 8, 10, 15, 20, 30],
            model_c="lr",
            model_t="lr",
            covariate_model_params={"cov_p": 0.35},
            est_params_c={"C": 100},
            est_params_t={"C": 1},
            effect_types=effect_types,
            model_names=["nv", "cf", "w", "west"],
            with_prop=True,
            treatment_assn_model=propensity_linear,
            treatment_model_params={"support_covs": [0]},
            file_name=file_name + "_setting1D",
            n_exp=n_exp,
            risk=True,
            save_file=True,
        )
    else:
        raise ValueError("Invalid setting name.")
    if return_res:
        return res
