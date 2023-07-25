# extract and save experiment results for comparing ConFair to OMN in their relationship between input degree and fairness improvement
# produce a single CSV file for all the settings
import warnings
import os
from PrepareData import read_json, make_folder

import argparse
from multiprocessing import Pool, cpu_count
import pandas as pd

warnings.filterwarnings('ignore')

def extract_evaluations_target(datasets, seeds, models, targets,
                            res_path='../intermediate/models/',
                            eval_path='eval/'):

    repo_dir = res_path.replace('intermediate/models/', '')
    eval_path = repo_dir + eval_path
    make_folder(eval_path)

    group_eval_metrics = ['AUC', 'ACC', 'SR', 'BalAcc', 'ERR', 'FPR', 'FNR', 'PR', 'TPR', 'TNR', 'TP', 'FN', 'TN', 'FP']
    overall_metrics = ['BalAcc', 'DI', 'EQDiff', 'AvgOddsDiff', 'SPDiff', 'FPRDiff', 'FNRDiff', 'ERRDiff']


    fair_weight_methods = ['scc', 'omn']
    fair_weight_bases = ['kam', 'one']

    # extract all the evaluation results for each dataset, model, and intervention method
    res_df = pd.DataFrame(columns=['data', 'model', 'seed', 'method', 'target', 'alpha_g1_y1', 'alpha_g1_y0', 'alpha_g0_y1', 'alpha_g0_y0', 'group', 'metric', 'value'])

    for data_name in datasets:
        cur_dir = res_path + data_name + '/'
        # get the results from model without intervention
        orig_df = pd.read_csv(eval_path + 'res-{}.csv'.format(data_name))
        for model_name in models:
            for seed in seeds:
                # get the results from model without intervention
                cur_orig_df = orig_df.query('data=="{}" and model=="{}" and seed=={} and method=="ORIG"'.format(data_name, model_name, seed))

                for weight_i, base_i in zip(fair_weight_methods, fair_weight_bases):
                    for target_flag in targets:
                        # get the results from model without intervention
                        cur_orig_df['target'] = target_flag
                        cur_orig_df['alpha_g1_y1'] = 0
                        cur_orig_df['alpha_g1_y0'] = 0
                        cur_orig_df['alpha_g0_y1'] = 0
                        cur_orig_df['alpha_g0_y0'] = 0

                        cur_orig_df = cur_orig_df[res_df.columns]
                        res_df = pd.concat([res_df, cur_orig_df])

                        if target_flag == 'DI':
                            if data_name == 'lsac':
                                degrees_pos = [x / 1000 for x in range(1, 401) if x % 2 == 0]
                            else:  # meps
                                degrees_pos = [x / 100 for x in range(1, 201) if x % 2 == 0]

                            if weight_i == 'scc':
                                degrees_g1_y1 = [1.0 for _ in range(len(degrees_pos))]
                                degrees_g1_y0 = [1.0 for _ in range(len(degrees_pos))]
                            elif weight_i == 'omn':
                                degrees_g1_y1 = [-x for x in degrees_pos]
                                degrees_g1_y0 = [x for x in degrees_pos]
                            else:
                                raise ValueError('Currently only support "reweigh_method" as ["scc", "omn"].')

                            degrees_g0_y1 = [x for x in degrees_pos]
                            degrees_g0_y0 = [-x for x in degrees_pos]
                            degrees_omn_lam = [x for x in degrees_pos]

                        elif target_flag == 'FNR':
                            # for error metric, the goal is to lower the value for the group who has higher values at beginning.
                            # For MEPS16, the majority group has higher FPR than the minority group.
                            # for lsac, the minority group (G0) has high FNR at the first place.
                            if data_name == 'meps16':
                                degrees_pos = [x / 100 for x in range(1, 401) if x % 2 == 0]

                                if weight_i == 'scc':
                                    degrees_g1_y0 = [1.0 for _ in range(len(degrees_pos))]
                                    degrees_g1_y1 = [x for x in degrees_pos]

                                    degrees_g0_y0 = [1.0 for _ in range(len(degrees_pos))]
                                    degrees_g0_y1 = [1.0 for _ in range(len(degrees_pos))]
                                elif weight_i == 'omn':
                                    degrees_g1_y1 = [x for x in degrees_pos]
                                    degrees_g0_y1 = [-x for x in degrees_pos]
                                    degrees_g1_y0 = [1.0 for _ in range(len(degrees_pos))]
                                    degrees_g0_y0 = [1.0 for _ in range(len(degrees_pos))]
                                    degrees_omn_lam = [x for x in degrees_pos]
                                else:
                                    raise ValueError('Currently only support "reweigh_method" as ["scc", "omn"].')
                            else:  # for lsac
                                degrees_pos = [x / 100 for x in range(1, 801) if x % 5 == 0]
                                if weight_i == 'scc':
                                    degrees_g1_y0 = [1.0 for _ in range(len(degrees_pos))]
                                    degrees_g1_y1 = [1.0 for _ in range(len(degrees_pos))]

                                    degrees_g0_y0 = [1.0 for _ in range(len(degrees_pos))]
                                    degrees_g0_y1 = [x for x in degrees_pos]
                                elif weight_i == 'omn':
                                    degrees_g1_y1 = [x for x in degrees_pos]
                                    degrees_g0_y1 = [-x for x in degrees_pos]

                                    degrees_g1_y0 = [1.0 for _ in range(len(degrees_pos))]
                                    degrees_g0_y0 = [1.0 for _ in range(len(degrees_pos))]
                                    degrees_omn_lam = [x for x in degrees_pos]
                                else:
                                    raise ValueError('Currently only support "reweigh_method" as ["scc", "omn"].')
                        elif target_flag == 'FPR':

                            degrees_pos = [x / 100 for x in range(1, 801) if x % 5 == 0]
                            # for error metric, the goal is to lower the value for the group who has higher values at beginning.
                            # For MEPS16 and lsac, the majority group has higher FPR than the minority group.
                            if weight_i == 'scc':
                                degrees_g1_y1 = [1.0 for _ in range(len(degrees_pos))]
                                degrees_g1_y0 = [x for x in degrees_pos]
                                degrees_g0_y1 = [1.0 for _ in range(len(degrees_pos))]
                                degrees_g0_y0 = [1.0 for _ in range(len(degrees_pos))]
                            elif weight_i == 'omn':
                                degrees_g1_y1 = [1.0 for _ in range(len(degrees_pos))]
                                degrees_g0_y1 = [1.0 for _ in range(len(degrees_pos))]
                                degrees_g1_y0 = [x for x in degrees_pos]
                                degrees_g0_y0 = [-x for x in degrees_pos]
                                degrees_omn_lam = [x for x in degrees_pos]
                            else:
                                raise ValueError('Currently only support "reweigh_method" as ["scc", "omn"].')
                        else:
                            raise ValueError('The input target is not supported. Choose from ["DI", "FNR", "FPR"].')

                        for alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0, omn_lam in zip(degrees_g1_y1, degrees_g1_y0, degrees_g0_y1, degrees_g0_y0, degrees_omn_lam):
                            if weight_i == 'scc':
                                weighing_output = '{}_{}_{}_{}_{}'.format(target_flag, alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0)
                            elif weight_i == 'omn':
                                weighing_output = '{}_{}'.format(target_flag, omn_lam)
                            else:
                                raise ValueError('Currently only support "reweigh_method" as ["scc", "omn"].')
                            eval_single_name = '{}eval-{}-{}-{}-{}-{}.json'.format(cur_dir, model_name, seed, weight_i, base_i, weighing_output)

                            method_name = weight_i.upper() + '-' + base_i.upper()

                            if os.path.exists(eval_single_name):
                                eval_res = read_json(eval_single_name)
                                for group in ['all', 'G0', 'G1']:
                                    base = [data_name, model_name.upper(), seed, method_name, target_flag, alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0, group]
                                    for metric_i in group_eval_metrics:
                                        res_df.loc[res_df.shape[0]] = base + [metric_i, eval_res[group][metric_i]]
                                for metric_i in overall_metrics:
                                    res_df.loc[res_df.shape[0]] = [data_name, model_name.upper(), seed, method_name, target_flag, alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0, 'all'] + [metric_i, eval_res['all'][metric_i]]
                            else:
                                print('--> Adding zero rows Because no eval for', eval_single_name)
                                for metric_i in overall_metrics:
                                    res_df.loc[res_df.shape[0]] = [data_name, model_name.upper(), seed, method_name, target_flag, alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0, 'all'] + [metric_i, 0]

    res_df.to_csv(eval_path+'degree-eval.csv', index=False)
    print('Result is saved at', eval_path+'degree-eval.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract evaluation results")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--data", type=str, default='all',
                        help="name of datasets over which the script is running. Default is for all the datasets.")
    parser.add_argument("--set_n", type=int, default=None,
                        help="number of datasets over which the script is running. Default is None for all the datasets.")
    parser.add_argument("--model", type=str, default='lr',
                        help="extract results for all the models as default. Otherwise, only extract the results for the input model from ['lr', 'tr'].")
    parser.add_argument("--target", type=str, default='all',
                        help="the target metric to optimize for ConFair. Choose from ['DI', 'FNR', 'FPR'].")

    parser.add_argument("--exec_n", type=int, default=3,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['meps16', 'lsac']
    seeds = [1, 12345, 6, 2211, 15]

    targets = ['DI', 'FNR', 'FPR']

    if args.target == 'all':
        pass
    elif args.target in targets:
        targets = [args.target]
    else:
        raise ValueError('The input "target" is not valid. Choose from ["DI", "FNR", "FPR"].')

    if args.exec_n is None:
        raise ValueError('The input "exec_n" is requried. Use "--exec_n 1" for a single execution.')
    elif type(args.exec_n) == str:
        raise ValueError('The input "exec_n" requires integer. Use "--exec_n 1" for a single execution.')
    else:
        n_exec = int(args.exec_n)
        seeds = seeds[:n_exec]

    if args.data == 'all':
        pass
    elif args.data in datasets:
        datasets = [args.data]
    elif 'seed' in args.data:
        datasets = [args.data]
    else:
        raise ValueError(
            'The input "data" is not valid. CHOOSE FROM ["seed#", "lsac", "cardio", "bank", "meps16", "credit", "ACSE", "ACSP", "ACSH", "ACSM", "ACSI"].')

    if args.set_n is not None:
        if type(args.set_n) == str:
            raise ValueError(
                'The input "set_n" requires integer. Use "--set_n 1" for running over a single dataset.')
        else:
            n_datasets = int(args.set_n)
            if n_datasets < 0:
                datasets = datasets[n_datasets:]
            elif n_datasets > 0:
                datasets = datasets[:n_datasets]
            else:
                raise ValueError(
                    'The input "set_n" requires non-zero integer. Use "--set_n 1" for running over a single dataset.')
    if args.model == 'all':
        models = ['lr', 'tr']
    elif args.model in ['lr', 'tr']:
        models = [args.model]
    else:
        raise ValueError('The input "model" is not valid. CHOOSE FROM ["lr", "tr"].')


    repo_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = repo_dir + '/intermediate/models/'

    if args.run == 'parallel':
        tasks = [[datasets, seeds, models, targets, res_path]]

        with Pool(cpu_count()//2) as pool:
            pool.starmap(extract_evaluations_target, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')