import argparse

import compcate.experiments.experiment_simulation
import compcate.experiments.experiment_twins

def init_arg():
    # arg parser if script is run from shell
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default='sim', type=str)
    parser.add_argument("--setting", default='1', type=str)
    parser.add_argument("--n_exp", default=10, type=int)
    parser.add_argument("--support_cov", default=True, type=bool)
    parser.add_argument("--model_t", default='cons', type=str)
    parser.add_argument('--file_name', default='', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = init_arg()
    if args.experiment == 'sim':
        compcate.experiments.experiment_simulation.run_experiment_by_setting(
            setting=args.setting, n_exp=args.n_exp, file_name=args.file_name)
    elif args.experiment == 'twins':
        compcate.experiments.experiment_twins.run_experiment_by_setting(
            setting=args.setting, n_exp=args.n_exp, get_support_covs_from_model=args.support_cov,
            model_t=args.model_t, file_name=args.file_name)
    else:
        raise ValueError("Unknown experiment name. Should be either `sim' or `twins'.")
