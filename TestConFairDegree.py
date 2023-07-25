# experiment code for comparing ConFair to OMN in their relationship between input degree and fairness improvement
# produce and save results for each method, dataset, fairness metric over LR models
import warnings
import argparse, os
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
from joblib import dump
from sklearn.metrics import confusion_matrix
from PrepareData import read_json, save_json
from TrainMLModels import LogisticRegression, XgBoost, generate_model_predictions, find_optimal_thres, compute_bal_acc
from EvaluateModels import eval_settings

warnings.filterwarnings(action='ignore')

def compute_metric(y_true, y_pred, metric='DI', label_order=[0, 1]):
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=label_order).ravel()
    pred_P = TP+FP
    P = TP + FN
    N = TN + FP
    if metric == 'DI':
        return pred_P / (P+N)
    elif metric == 'FPR':
        fpr = FP / N if N > 0.0 else np.float64(0.0)
        return fpr
    elif metric == 'FNR':
        fnr = FN / P if P > 0.0 else np.float64(0.0)
        return fnr
    else:
        raise ValueError('The input of parameter "metric" is not supported. CHOSSE FROM ["DI", "FPR", "FNR"].')

def eval_target_diff(test_eval_df, pred_col, target='DI', sensi_col='A', n_groups=2):
    metric_all = []
    for group_i in range(n_groups): # 0 is minority group
        group_df = test_eval_df[test_eval_df[sensi_col] == group_i]
        group_value = compute_metric(group_df['Y'], group_df[pred_col], metric=target)
        metric_all.append(group_value)

    return metric_all[0] - metric_all[1]

def compute_weights_flexible(df, method, sample_base='kam',
                    alpha_g1_y1 = -1, alpha_g1_y0 = 1, alpha_g0_y1 = 2, alpha_g0_y0 = 1,
                    omn_lam=1.0, omn_target='DI',
                    cc_par=None, cc_col='vio_cc', cc_vio_thres=0.1,
                    sensi_col='A', y_col='Y'):

    group_1_y_1 = np.array(df[sensi_col] == 1).astype(int) * np.array(df[y_col] == 1).astype(int)
    group_1_y_0 = np.array(df[sensi_col] == 1).astype(int) * np.array(df[y_col] == 0).astype(int)
    group_0_y_1 = np.array(df[sensi_col] == 0).astype(int) * np.array(df[y_col] == 1).astype(int)
    group_0_y_0 = np.array(df[sensi_col] == 0).astype(int) * np.array(df[y_col] == 0).astype(int)
    group_1 = np.array(df[sensi_col] == 1).astype(int)
    group_0 = np.array(df[sensi_col] == 0).astype(int)
    target_1 = np.array(df[y_col] == 1).astype(int)
    target_0 = np.array(df[y_col] == 0).astype(int)

    if method == 'scc':
        if cc_par is not None: # if mean(violation) > 0.1, use the corresponding zero violations in deciding weights
            group_1_y_1_mean = int(cc_par['mean_train_G1_L1'] >= cc_vio_thres)
            group_1_y_0_mean = int(cc_par['mean_train_G1_L0'] >= cc_vio_thres)
            group_0_y_1_mean = int(cc_par['mean_train_G0_L1'] >= cc_vio_thres)
            group_0_y_0_mean = int(cc_par['mean_train_G0_L0'] >= cc_vio_thres)
        else:
            group_1_y_1_mean = 1
            group_1_y_0_mean = 1
            group_0_y_1_mean = 1
            group_0_y_0_mean = 1

        group_1_y_1_vio_0 = np.array(df[sensi_col] == 1).astype(int) * np.array(df[y_col] == 1).astype(int) * np.array(df[cc_col] == 0).astype(int)
        group_1_y_0_vio_0 = np.array(df[sensi_col] == 1).astype(int) * np.array(df[y_col] == 0).astype(int) * np.array(df[cc_col] == 0).astype(int)
        group_0_y_1_vio_0 = np.array(df[sensi_col] == 0).astype(int) * np.array(df[y_col] == 1).astype(int) * np.array(df[cc_col] == 0).astype(int)
        group_0_y_0_vio_0 = np.array(df[sensi_col] == 0).astype(int) * np.array(df[y_col] == 0).astype(int) * np.array(df[cc_col] == 0).astype(int)

    total_n = df.shape[0]
    if method == 'scc': # default kam to balance POS between groups
        if sample_base == 'kam':
            sample_weights = np.zeros(total_n)

            sample_weights += group_1_y_1 * (np.sum(group_1) * np.sum(target_1) / (total_n * np.sum(group_1_y_1))) \
                              + group_1_y_0 * (np.sum(group_1) * np.sum(target_0) / (total_n * np.sum(group_1_y_0))) \
                              + group_0_y_1 * (np.sum(group_0) * np.sum(target_1) / (total_n * np.sum(group_0_y_1))) \
                              + group_0_y_0 * (np.sum(group_0) * np.sum(target_0) / (total_n * np.sum(group_0_y_0)))
        elif sample_base == 'omn': # default for target as DI
            sample_weights = np.ones(total_n)
            sample_weights -= omn_lam * total_n / np.sum(group_1) * group_1_y_1 \
                              - omn_lam * total_n / np.sum(group_1) * group_1_y_0 \
                              - omn_lam * total_n / np.sum(group_0) * group_0_y_1 \
                              + omn_lam * total_n / np.sum(group_0) * group_0_y_0

        elif sample_base == 'zero':
            sample_weights = np.zeros(total_n)
        elif sample_base == 'one':
            sample_weights = np.ones(total_n)
        else:
            raise ValueError('The input sample_base parameter is not supported. Choose from "[kam, omn, zero, one]".')

        # intervention four weighing parameters that are with signs to represent direction: alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0
        sample_weights += alpha_g1_y1 * group_1_y_1_mean * group_1_y_1_vio_0 \
                          + alpha_g1_y0 * group_1_y_0_mean * group_1_y_0_vio_0 \
                          + alpha_g0_y1 * group_0_y_1_mean * group_0_y_1_vio_0 \
                          + alpha_g0_y0 * group_0_y_0_mean * group_0_y_0_vio_0

    elif method == 'kam':
        sample_weights = np.zeros(total_n)

        sample_weights += group_1_y_1 * (np.sum(group_1) * np.sum(target_1) / (total_n * np.sum(group_1_y_1))) \
                          + group_1_y_0 * (np.sum(group_1) * np.sum(target_0) / (total_n * np.sum(group_1_y_0))) \
                          + group_0_y_1 * (np.sum(group_0) * np.sum(target_1) / (total_n * np.sum(group_0_y_1))) \
                          + group_0_y_0 * (np.sum(group_0) * np.sum(target_0) / (total_n * np.sum(group_0_y_0)))
    elif method == 'omn': # version for three fairness targets
        sample_weights = np.ones(total_n)
        if omn_target == 'DI':
            sample_weights -= omn_lam * total_n / np.sum(group_1) * group_1_y_1 \
                              - omn_lam * total_n / np.sum(group_1) * group_1_y_0 \
                              - omn_lam * total_n / np.sum(group_0) * group_0_y_1 \
                              + omn_lam * total_n / np.sum(group_0) * group_0_y_0
        elif omn_target == 'FNR':
            sample_weights += omn_lam * total_n / np.sum(group_1_y_1) * group_1_y_1 \
                              - omn_lam * total_n / np.sum(group_0_y_1) * group_0_y_1

        elif omn_target == 'FPR':
            sample_weights += omn_lam * total_n / np.sum(group_1_y_0) * group_1_y_0 \
                              - omn_lam * total_n / np.sum(group_0_y_0) * group_0_y_0
        else:
            raise ValueError('The input "omn_target" parameter is not supported. Choose from "[DI, FNR, FPR]".')
    else:
        raise ValueError('The input method parameter is not supported. Choose from "[kam, omn, scc]".')

    return sample_weights



def retrain_ML_models_four_degrees(data_name, seed, model_name, reweigh_method, weight_base, target_flag,
                                  alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0, omn_lam,
                                  res_path='../intermediate/models/',
                                  data_path='data/processed/', y_col = 'Y', sensi_col='A'):

    cur_dir = res_path + data_name + '/'
    repo_dir = res_path.replace('intermediate/models/', '')
    # default CC OPT is true
    if model_name == 'tr':
        train_df = pd.read_csv('{}train-{}-bin.csv'.format(cur_dir, seed))
        val_df = pd.read_csv('{}val-{}-bin.csv'.format(cur_dir, seed))
        test_df = pd.read_csv('{}test-{}-bin.csv'.format(cur_dir, seed))

        vio_train_df = pd.read_csv('{}train-cc-{}.csv'.format(cur_dir, seed))
        train_df['vio_cc'] = vio_train_df['vio_cc']

    else:
        train_df = pd.read_csv('{}train-cc-{}.csv'.format(cur_dir, seed))
        val_df = pd.read_csv('{}val-cc-{}.csv'.format(cur_dir, seed))
        test_df = pd.read_csv('{}test-cc-{}.csv'.format(cur_dir, seed))

    meta_info = read_json('{}/{}{}.json'.format(repo_dir, data_path, data_name))

    cc_par = read_json('{}par-cc-{}.json'.format(cur_dir, seed))
    feature_setting = read_json('{}par-{}-{}.json'.format(cur_dir, model_name, seed))['model_setting']

    n_features = meta_info['n_features']  # including sensitive column

    if feature_setting == 'S1':
        features = ['X{}'.format(i) for i in range(1, n_features)] + [sensi_col]
    else:
        features = ['X{}'.format(i) for i in range(1, n_features)]

    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])
    val_data = val_df[features]

    weights = compute_weights_flexible(train_df, reweigh_method, weight_base, alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0, omn_lam, target_flag,
                                       cc_par=cc_par)
    if model_name == 'tr':
        learner = XgBoost()
    else:
        learner = LogisticRegression()

    model = learner.fit(train_data, Y_train, features, seed, weights)

    if reweigh_method == 'scc':
        weighing_output = '{}_{}_{}_{}_{}'.format(target_flag, alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0)
    elif reweigh_method == 'omn':
        weighing_output = '{}_{}'.format(target_flag, omn_lam)
    else:
        raise ValueError('Currently only support "reweigh_method" as ["scc", "omn"].')


    if model is not None:
        val_predict = generate_model_predictions(model, val_data)

        if sum(val_predict) == 0:
            print('==> model predict only one label for val data ', data_name, model_name, seed, reweigh_method, weight_base)

        val_df['Y_pred_scores'] = val_predict
        opt_thres = find_optimal_thres(val_df, opt_obj='BalAcc', num_thresh=100)
        cur_thresh = opt_thres['thres']

        val_df['Y_pred'] = val_df['Y_pred_scores'].apply(lambda x: int(x > cur_thresh))

        cur_diff = eval_target_diff(val_df, 'Y_pred', target_flag)
        cur_acc = compute_bal_acc(val_df['Y'], val_df['Y_pred'])

        if reweigh_method == 'scc':
            res_dict = {'BalAcc': cur_acc, 'thres': cur_thresh, 'degree': [alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0], '{}Diff'.format(target_flag): cur_diff}
        elif reweigh_method == 'omn':
            res_dict = {'BalAcc': cur_acc, 'thres': cur_thresh, 'degree': [omn_lam], '{}Diff'.format(target_flag): cur_diff}
        else:
            raise ValueError('Currently only support "reweigh_method" as ["scc", "omn"].')

        save_json(res_dict, '{}par-{}-{}-{}-{}-{}.json'.format(cur_dir, model_name, seed, reweigh_method, weight_base, weighing_output))

        test_data = test_df[features]
        test_predict = generate_model_predictions(model, test_data, cur_thresh)
        if sum(test_predict) == 0:
            print('==> model predict only one label for test data ', data_name, model_name, seed, reweigh_method, weight_base)
        test_df['Y_pred'] = test_predict

        dump(model,
             '{}{}-{}-{}-{}-{}.joblib'.format(cur_dir, model_name, seed, reweigh_method, weight_base, weighing_output))

        test_df[[sensi_col, 'Y', 'Y_pred']].to_csv(
            '{}pred-{}-{}-{}-{}-{}.csv'.format(cur_dir, model_name, seed, reweigh_method, weight_base, weighing_output),
            index=False)

        eval_res = eval_settings(test_df, sensi_col, 'Y_pred')
        save_json(eval_res, '{}eval-{}-{}-{}-{}-{}.json'.format(cur_dir, model_name, seed, reweigh_method, weight_base, weighing_output))

    else:
        print('no model fitted ', data_name, model_name, seed, reweigh_method, weight_base, weighing_output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test different targets and intervention degrees on real data")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--data", type=str, default='all',
                        help="name of datasets over which the script is running. Default is for all the datasets.")
    parser.add_argument("--set_n", type=int, default=None,
                        help="number of datasets over which the script is running. Default is 10.")
    parser.add_argument("--model", type=str, default='lr',
                        help="extract results for all the models as default. Otherwise, only extract the results for the input model from ['lr', 'tr'].")

    parser.add_argument("--weight", type=str, default='omn',
                        help="which weighing method to use in training ML models.")
    parser.add_argument("--base", type=str, default='one',
                        help="which base weights to combine in computing the final weights. For scc+kam and scc+omn, set the base as kam or omn.")

    parser.add_argument("--target", type=str, default='all',
                        help="the target metric to optimize for ConFair. Choose from ['DI', 'FNR', 'FPR'].")

    parser.add_argument("--exec_n", type=int, default=3,
                        help="number of executions with different random seeds. Default is 20.")
    args = parser.parse_args()

    datasets = ['meps16', 'lsac']
    seeds = [1, 12345, 6, 2211, 15]
    models = ['lr', 'tr']

    targets = ['DI', 'FNR', 'FPR']

    if args.target == 'all':
        pass
    elif args.target in targets:
        targets = [args.target]
    else:
        raise ValueError('The input "target" is not valid. Choose from ["DI", "FNR", "FPR"].')

    if args.data == 'all':
        pass
    elif args.data in datasets:
        datasets = [args.data]
    elif 'seed' in args.data:
        datasets = [args.data]
    else:
        raise ValueError(
            'The input "data" is not valid. CHOOSE FROM ["lsac", "meps16"].')

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
    if args.model in models:
        pass
    else:
        raise ValueError('The input "model" is not valid. CHOOSE FROM ["lr", "tr"].')

    if args.exec_n is None:
        raise ValueError(
            'The input "exec_n" is requried. Use "--exec_n 1" for a single execution.')
    elif type(args.exec_n) == str:
        raise ValueError(
            'The input "exec_n" requires integer. Use "--exec_n 1" for a single execution.')
    else:
        n_exec = int(args.exec_n)
        seeds = seeds[:n_exec]

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = repo_dir + '/intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            for seed in seeds: # for all metric, the implicit assumption is that the minority group has lower value than the majority group, the goal is to increase the metric value
                for target_flag in targets:
                    if target_flag == 'DI':
                        if data_name == 'lsac':
                            degrees_pos = [x / 1000 for x in range(1, 401) if x % 2 == 0]
                        else: # meps
                            degrees_pos = [x / 100 for x in range(1, 201) if x % 2 == 0]

                        degrees_g1_y1 = [1.0 for _ in range(len(degrees_pos))]
                        degrees_g1_y0 = [1.0 for _ in range(len(degrees_pos))]

                        degrees_g0_y1 = [x for x in degrees_pos]
                        degrees_g0_y0 = [-x for x in degrees_pos]

                        degrees_omn_lam = [x for x in degrees_pos]

                    elif target_flag == 'FNR':
                        # for error metric, the goal is to lower the value for the group who has higher values at beginning.
                        # For MEPS16, the majority group has higher FNR than the minority group.
                        # for lsac, the minority group (G0) has high FNR at the first place.
                        if data_name == 'meps16':
                            degrees_pos = [x / 100 for x in range(1, 401) if x % 2 == 0]

                            degrees_g1_y0 = [1.0 for _ in range(len(degrees_pos))]
                            degrees_g1_y1 = [x for x in degrees_pos]

                            degrees_g0_y0 = [1.0 for _ in range(len(degrees_pos))]
                            degrees_g0_y1 = [1.0 for _ in range(len(degrees_pos))]

                            degrees_omn_lam = [x for x in degrees_pos]
                        else: # for lsac
                            degrees_pos = [x / 100 for x in range(1, 801) if x % 5 == 0]

                            degrees_g1_y0 = [1.0 for _ in range(len(degrees_pos))]
                            degrees_g1_y1 = [1.0 for _ in range(len(degrees_pos))]

                            degrees_g0_y0 = [1.0 for _ in range(len(degrees_pos))]
                            degrees_g0_y1 = [x for x in degrees_pos]

                            degrees_omn_lam = [x for x in degrees_pos]
                    elif target_flag == 'FPR':
                        # for error metric, the goal is to lower the value for the group who has higher values at beginning.
                        # For MEPS16 and lsac, the majority group has higher FPR than the minority group.

                        degrees_pos = [x / 100 for x in range(1, 801) if x % 5 == 0]

                        degrees_g1_y1 = [1.0 for _ in range(len(degrees_pos))]
                        degrees_g1_y0 = [x for x in degrees_pos]


                        degrees_g0_y1 = [1.0 for _ in range(len(degrees_pos))]
                        degrees_g0_y0 = [1.0 for _ in range(len(degrees_pos))]

                        degrees_omn_lam = [x for x in degrees_pos]

                    else:
                        raise ValueError('The input target is not supported. Choose from ["DI", "FNR", "FPR"].')

                    for alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0, omn_lam in zip(degrees_g1_y1, degrees_g1_y0, degrees_g0_y1, degrees_g0_y0, degrees_omn_lam):
                        tasks.append([data_name, seed, args.model, args.weight, args.base, target_flag, alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0, omn_lam, res_path])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(retrain_ML_models_four_degrees, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')