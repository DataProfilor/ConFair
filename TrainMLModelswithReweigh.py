# Train ML models with reweigh samples by different methods
import warnings
import timeit
import argparse, os
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from joblib import dump, load
from PrepareData import read_json, save_json
from TrainMLModels import LogisticRegression, XgBoost, generate_model_predictions, find_optimal_thres
from TestConFairDegree import compute_weights_flexible, eval_target_diff

from copy import deepcopy

warnings.filterwarnings(action='ignore')

def scc_transform_degrees(input_degree, target, reverse_flag=False):
    # transform the input degree to the required interventions according to the input target
    if target == 'DI':
        if reverse_flag:  # default assumption is to intervene on minority group G0, if reverse_flag is true, intervene on G1
            alpha_g1_y1 = input_degree
            alpha_g1_y0 = -input_degree

            alpha_g0_y1 = 1.0
            alpha_g0_y0 = 1.0
        else:
            alpha_g1_y1 = 1.0
            alpha_g1_y0 = 1.0

            alpha_g0_y1 = input_degree
            alpha_g0_y0 = -input_degree

    elif target == 'FNR':
        if reverse_flag:
            alpha_g1_y1 = input_degree
            alpha_g0_y1 = 1.0
        else:
            alpha_g1_y1 = 1.0
            alpha_g0_y1 = input_degree

        alpha_g1_y0 = 1.0
        alpha_g0_y0 = 1.0

    elif target == 'FPR':
        if reverse_flag:
            alpha_g1_y0 = input_degree
            alpha_g0_y0 = 1.0
        else:
            alpha_g1_y0 = 1.0
            alpha_g0_y0 = input_degree

        alpha_g1_y1 = 1.0
        alpha_g0_y1 = 1.0
    else:
        raise ValueError('SCC TRANSFORM: The input "target" is not supported. Choose from ["DI", "FNR", "FPR"].')
    return (alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0)

def get_metric(settings, inter_degree, reverse_flag=False, y_col = 'Y'):
    learner, train_df, features, validate_df, reweigh_method, weight_base, target, seed, model_name, data_name, cc_par = settings

    if reweigh_method == 'scc':
        alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0 = scc_transform_degrees(inter_degree, target, reverse_flag)
        weights = compute_weights_flexible(train_df, reweigh_method, weight_base,
                                            alpha_g1_y1 = alpha_g1_y1, alpha_g1_y0 = alpha_g1_y0, alpha_g0_y1 = alpha_g0_y1, alpha_g0_y0 = alpha_g0_y0,
                                            cc_par=cc_par)
    elif reweigh_method == 'omn':
        weights = compute_weights_flexible(train_df, reweigh_method, weight_base,
                                           omn_lam=inter_degree, omn_target=target)
    elif reweigh_method == 'kam':
        weights = compute_weights_flexible(train_df, reweigh_method, weight_base)
    else:
        raise ValueError('==> GET metric: The input method parameter is not supported. Choose from "[kam, omn, scc]".')

    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])

    model = learner.fit(train_data, Y_train, features, seed, weights)
    if model is not None:
        val_data = validate_df[features]
        val_predict = generate_model_predictions(model, val_data)
        if sum(val_predict) == 0:
            print('==> GET metric: only one label for val set at \n', data_name, model_name, seed, reweigh_method, weight_base, target)

        validate_df['Y_pred_scores'] = val_predict
        opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc', num_thresh=100)
        cur_thresh = opt_thres['thres']

        validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > cur_thresh))
        cur_diff = eval_target_diff(validate_df, 'Y_pred', target)
        cur_acc = opt_thres['BalAcc']
        return (model, cur_thresh, cur_diff, cur_acc)
    else:
        print('++ GET metric with no model fitted for train set at \n', data_name, seed, model_name, reweigh_method, weight_base, target, inter_degree)
        return None

def exec_degrees_scc(settings, degrees, reverse_flag=False, max_diff=0.001):
    best_degree = None
    best_metric = 10
    best_thres = None
    best_model = None
    best_acc = None

    for degree_try in degrees:
        res_try = get_metric(settings, degree_try, reverse_flag=reverse_flag)

        if res_try is not None:
            model_try, thres_try, metric_try, acc_try = res_try
            # print('==>try', degree_try, sp_try)
            if abs(metric_try) < max_diff and acc_try > 0.5:
                best_degree = degree_try
                best_metric = metric_try
                best_thres = thres_try
                best_model = model_try
                best_acc = acc_try
                break # done with search
            elif abs(metric_try) <= abs(best_metric) and abs(metric_try) > 0: # record and continue to next degree
                best_degree = degree_try
                best_metric = metric_try
                best_thres = thres_try
                best_model = model_try
                best_acc = acc_try

    return (best_model, best_thres, best_degree, best_metric, best_acc)

def search_opt_degree_scc(settings, degrees_list=[x/10 for x in range(1, 21)], init_metric=None, count_iter=1, max_diff=0.001, reverse_flag=False):
    init_metric = init_metric or 1
    best_model, best_thres, best_degree, best_metric, best_acc = exec_degrees_scc(settings, degrees_list, reverse_flag=reverse_flag)
    if best_degree is None:
        print('==> SCC search: returned best_degree is none at \n', settings[4:10])
        return None
    elif abs(best_metric) <= max_diff: # search stops according to the condition of max_diff
        return (best_model, best_thres, best_degree, best_metric, best_acc)
    elif abs(init_metric) - abs(best_metric) <= 0.0001: # search stops because the metric stops to improve
        return (best_model, best_thres, best_degree, best_metric, best_acc)
    else: #search continue to finite degrees by dividing the current degree with 100*count_iter
        # estimate the boundary of the new search space of the degrees
        left_try = max(0.01/count_iter, best_degree - 0.05/count_iter)
        right_try = best_degree + 0.05/count_iter
        left_res = get_metric(settings, left_try, reverse_flag=reverse_flag)
        right_res = get_metric(settings, right_try, reverse_flag=reverse_flag)

        middle_bound = int(best_degree * 100 * count_iter)

        if left_res is not None and right_res is not None: # refine the boundary by the metric value
            _, _, metric_left, _ = left_res
            _, _, metric_right, _ = right_res
            left_degree = max(0.01/count_iter, best_degree - 0.1 / count_iter)
            right_degree = best_degree + 0.1 / count_iter
            left_bound = int(left_degree*100*count_iter)
            right_bound = int(right_degree*100*count_iter)

            if abs(metric_left) < abs(metric_right):
                next_degrees = [x/100/count_iter for x in range(left_bound, middle_bound)]
            else:
                next_degrees = [x/100/count_iter for x in range(middle_bound, right_bound)]

        elif left_res is not None:
            left_degree = max(0.01/count_iter, best_degree - 0.1 / count_iter)
            left_bound = int(left_degree * 100 * count_iter)
            next_degrees = [x/100/count_iter for x in range(left_bound, middle_bound)]
        elif right_res is not None:
            right_degree = best_degree + 0.1 / count_iter
            right_bound = int(right_degree * 100 * count_iter)
            next_degrees = [x/100/count_iter for x in range(middle_bound, right_bound)]
        else:
            raise ValueError('==> SCC search: Left and right temp degrees are both invalid at \n', settings[4:10])
        return search_opt_degree_scc(settings, next_degrees, init_metric=best_metric, count_iter=count_iter*10)

def search_opt_degree_omn(settings, low=0, high=2, epsilon = 0.02, y_col='Y'):
    learner, train_df, features, validate_df, reweigh_method, weight_base, target, seed, model_name, cur_path, _ = settings

    best_acc = -1
    best_diff = 1
    termination_flag = False

    init_model = load('{}{}-{}.joblib'.format(cur_path, model_name, seed))
    init_thresh = read_json('{}par-{}-{}.json'.format(cur_path, model_name, seed))['thres']
    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])

    val_data = validate_df[features]
    val_predict = generate_model_predictions(init_model, val_data)
    if sum(val_predict) == 0:
        print('==> OMN search: predict only one label for val set at \n', cur_path, model_name, seed, reweigh_method, weight_base, target)

    validate_df['Y_pred_scores'] = val_predict
    validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > init_thresh))
    # print('TEST ', target, '______')
    init_diff = eval_target_diff(validate_df, 'Y_pred', target=target)

    best_model = None
    best_degree = 0
    best_threshold = 0

    cur_diff = init_diff

    while (abs(cur_diff) >= epsilon or not termination_flag) and (high - low > 0.0001):
        mid = (high + low) / 2
        weights = compute_weights_flexible(train_df, reweigh_method, weight_base, omn_lam=mid, omn_target=target)

        cur_model = learner.fit(train_data, Y_train, features, seed, weights)
        if cur_model is not None:
            val_predict = generate_model_predictions(cur_model, val_data)
            if sum(val_predict) == 0:
                print('==> OMN search in while: predict only one label for val set at \n', cur_path, model_name, seed, reweigh_method, weight_base, target)
            validate_df['Y_pred_mid'] = val_predict

            model_prob = validate_df['Y_pred_mid'].unique()
            if len(model_prob) > 5: # reasonable probability outputted
                opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc', pred_col='Y_pred_mid', num_thresh=100)
                cur_acc = opt_thres['BalAcc']
                cur_thresh = opt_thres['thres']

                validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > cur_thresh))
                cur_diff = eval_target_diff(validate_df, 'Y_pred', target=target)

            else:
                print('++ OMN search: no model fitted at \n', seed, model_name, reweigh_method, weight_base, target, mid)
                cur_diff = 1
                cur_acc = -1
        else: # no reasonable probability outputted
            print('++ OMN search: no model fitted at \n', cur_path, seed, model_name, reweigh_method, weight_base, target, mid)
            cur_diff = 1
            cur_acc = -1

        if (init_diff > 0 and cur_diff < epsilon) or (init_diff < 0 and cur_diff > -1 * epsilon):
            high = mid
        else:
            low = mid

        if abs(cur_diff) <= epsilon: # stop search accroding to epsilon (max difference)
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_diff = cur_diff
                best_model = deepcopy(cur_model)
                best_degree = mid
                best_threshold = cur_thresh
                # print('!!! OMN search: {} {} satisfied at degree {} with target {} diff {} acc {} ---'.format(model_name, seed, mid, target, best_diff, best_acc))
                break
        else: # continue search with update intermediate results first
            if abs(cur_diff) < abs(best_diff):
                best_acc = cur_acc
                best_diff = cur_diff
                best_model = deepcopy(cur_model)
                best_degree = mid
                best_threshold = cur_thresh

    if best_threshold == 0: # the trained model is not reasonable with outputting nonsense probability according to the recorded threshold
        return None
    else:
        return (best_model, best_threshold, best_degree, best_diff, best_acc)

def retrain_model_with_degree(settings, input_degree, reverse_flag=False, y_col='Y'):
    learner, train_df, features, validate_df, reweigh_method, weight_base, target, seed, model_name, data_name, cc_par = settings
    train_data = train_df[features]
    Y_train = np.array(train_df[y_col])

    if reweigh_method == 'scc':
        alpha_g1_y1, alpha_g1_y0, alpha_g0_y1, alpha_g0_y0 = scc_transform_degrees(input_degree, target, reverse_flag)
        weights = compute_weights_flexible(train_df, reweigh_method, weight_base,
                                            alpha_g1_y1 = alpha_g1_y1, alpha_g1_y0 = alpha_g1_y0, alpha_g0_y1 = alpha_g0_y1, alpha_g0_y0 = alpha_g0_y0,
                                            cc_par=cc_par)
    elif reweigh_method == 'omn':
        weights = compute_weights_flexible(train_df, reweigh_method, weight_base,
                                           omn_lam=input_degree, omn_target=target)
    elif reweigh_method == 'kam':
        weights = compute_weights_flexible(train_df, reweigh_method, weight_base)

    elif reweigh_method == 'cap': # for CAP, using no weights version
        weights = None
    else:
        raise ValueError('==> RETRAIN with degree: The input method parameter is not supported. Choose from "[kam, omn, scc, cap]".')


    best_model = learner.fit(train_data, Y_train, features, seed, weights)


    if best_model is not None:
        val_data = validate_df[features]
        val_predict  = generate_model_predictions(best_model, val_data)
        if sum(val_predict) == 0:
            print('==> RETRAIN with degree: predict only one label for val set at \n', data_name, model_name, seed, reweigh_method, weight_base, target)
        validate_df['Y_pred_scores'] = val_predict

        opt_thres = find_optimal_thres(validate_df, opt_obj='BalAcc', num_thresh=100)
        best_threshold = opt_thres['thres']

        validate_df['Y_pred'] = validate_df['Y_pred_scores'].apply(lambda x: int(x > best_threshold))
        best_diff = eval_target_diff(validate_df, 'Y_pred', target)
        best_acc = opt_thres['BalAcc']
        return (best_model, best_threshold, input_degree, best_diff, best_acc)
    else:
        print('++ RETRAIN with degree: no model fitted for train set at \n', data_name, seed, model_name, reweigh_method, weight_base, input_degree, target)
        return None


def run_retrain_model(data_name, seed, model_name, reweigh_method, weight_base, target='DI', input_degree=None,
                                    res_path='../intermediate/models/', cc_opt=True, reverse_flag=False,
                                    data_path='data/processed/', sensi_col='A'):

    repo_dir = res_path.replace('intermediate/models/', '')
    cur_dir = res_path + data_name + '/'

    if cc_opt: # whether to run the optimization of CC
        opt_suffix = ''
    else:
        opt_suffix = '-noOPT'

    if model_name == 'tr':
        if reweigh_method == 'cap':
            train_df = pd.read_csv('{}train-{}-bin-{}.csv'.format(cur_dir, seed, reweigh_method))
        else:
            train_df = pd.read_csv('{}train-{}-bin.csv'.format(cur_dir, seed))

        validate_df = pd.read_csv('{}val-{}-bin.csv'.format(cur_dir, seed))
        test_df = pd.read_csv('{}test-{}-bin.csv'.format(cur_dir, seed))

        vio_train_df = pd.read_csv('{}train-cc-{}{}.csv'.format(cur_dir, seed, opt_suffix))
        train_df['vio_cc'] = vio_train_df['vio_cc']
        learner = XgBoost()
    else:
        train_df = pd.read_csv('{}train-cc-{}{}.csv'.format(cur_dir, seed, opt_suffix))
        validate_df = pd.read_csv('{}val-cc-{}{}.csv'.format(cur_dir, seed, opt_suffix))
        test_df = pd.read_csv('{}test-cc-{}{}.csv'.format(cur_dir, seed, opt_suffix))
        learner = LogisticRegression()

    meta_info = read_json('{}/{}{}.json'.format(repo_dir, data_path, data_name))
    feature_setting = read_json('{}par-{}-{}.json'.format(cur_dir, model_name, seed))['model_setting']
    n_features = meta_info['n_features']

    if feature_setting == 'S1': # including sensitive column in the training
        features = ['X{}'.format(i) for i in range(1, n_features)] + [sensi_col]
    else:
        features = ['X{}'.format(i) for i in range(1, n_features)]

    if reweigh_method == 'kam':
        start = timeit.default_timer()
        settings = (learner, train_df, features, validate_df, reweigh_method, weight_base, target, seed, model_name, data_name, None)
        res = retrain_model_with_degree(settings, 0)

    elif reweigh_method == 'omn': # integrate the code from OmniFair
        start = timeit.default_timer()
        if input_degree: # user-specified intervention degree
            settings = (learner, train_df, features, validate_df, reweigh_method, weight_base, target, seed, model_name, data_name, None)
            res = retrain_model_with_degree(settings, input_degree, reverse_flag)
        else: # search for the optimal degree automatically
            settings = (learner, train_df, features, validate_df, reweigh_method, weight_base, target, seed, model_name, cur_dir, None)
            res = search_opt_degree_omn(settings)

    elif reweigh_method == 'scc':
        start = timeit.default_timer()
        cc_par = read_json('{}par-cc-{}{}.json'.format(cur_dir, seed, opt_suffix))
        settings = (learner, train_df, features, validate_df, reweigh_method, weight_base, target, seed, model_name, data_name, cc_par)
        if input_degree: # user-specified intervention degree
            res = retrain_model_with_degree(settings, input_degree, reverse_flag)
        else: # search for the optimal degree automatically
            res = search_opt_degree_scc(settings, reverse_flag=reverse_flag)

    elif reweigh_method == 'cap':
        start = timeit.default_timer()
        settings = (learner, train_df, features, validate_df, reweigh_method, weight_base, target, seed, model_name, data_name, None)
        res = retrain_model_with_degree(settings, 0)
    else:
        raise ValueError('RUN RETRAIN: Not supported methods! CHOOSE FROM [scc, omn, kam, cap].')


    end = timeit.default_timer()
    time = end - start

    if res is not None:
        best_model, best_threshold, best_degree, best_diff, best_acc = res
        res_dict = {'time': time, 'BalAcc': best_acc, 'thres': best_threshold, 'degree': best_degree, '{}Diff'.format(target): best_diff}
        save_json(res_dict, '{}par-{}-{}-{}-{}-{}{}.json'.format(cur_dir, model_name, seed, reweigh_method, weight_base, target, opt_suffix))

        test_data = test_df[features]
        test_predict = generate_model_predictions(best_model, test_data, best_threshold)
        if sum(test_predict) == 0:
            print('==> RUN RETRAIN: predict only one label for test set at \n', data_name, model_name, seed, reweigh_method, weight_base, target)
        test_df['Y_pred'] = test_predict

        dump(best_model, '{}{}-{}-{}-{}-{}{}.joblib'.format(cur_dir, model_name, seed, reweigh_method, weight_base, target, opt_suffix))
        test_df[[sensi_col, 'Y', 'Y_pred']].to_csv('{}pred-{}-{}-{}-{}-{}{}.csv'.format(cur_dir, model_name, seed, reweigh_method, weight_base, target, opt_suffix), index=False)
    else:
        print('++ RUN RETRAIN: No model is found for \n', data_name, seed, model_name, reweigh_method, weight_base, target)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Retrain ML models with fair proprocessing methods")
    parser.add_argument("--run", type=str, default='parallel',
                        help="setting of 'parallel' for system evaluation or 'serial' execution for unit test.")
    parser.add_argument("--data", type=str, default='all',
                        help="name of datasets over which the script is running. Default is for all the datasets.")
    parser.add_argument("--set_n", type=int, default=None,
                        help="number of datasets over which the script is running. Default is None for all the datasets.")
    parser.add_argument("--model", type=str, default='all',
                        help="extract results for all the models as default. Otherwise, only extract the results for the input model from ['lr', 'tr'].")
    parser.add_argument("--weight", type=str, default='scc',
                        help="which weighing method to use in training ML models.")
    parser.add_argument("--base", type=str, default='kam',
                        help="which base weights to combine in computing the final weights. For scc+kam and scc+omn, set the base as kam or omn.")
    parser.add_argument("--degree", type=float, default=None,
                        help="additional weights in OmniFair and ConFair. Default is None for all the datasets and will be searched for optimal value automatically.")

    parser.add_argument("--target", type=str, default='DI',
                        help="the target metric to optimize for ConFair. Choose from ['DI', 'FNR', 'FPR'].")

    parser.add_argument("--exec_n", type=int, default=20,
                        help="number of executions with different random seeds. Default is 20.")

    parser.add_argument("--opt", type=int, default=1,
                        help="whether to apply the optimization for CC tool.")

    args = parser.parse_args()

    all_supported_data = ['meps16', 'lsac', 'ACSP', 'credit', 'ACSE', 'ACSH', 'ACSI'] + ['seed{}'.format(x) for x in
                                                                                         [12345, 15, 433, 57005, 7777]]
    if args.data == 'all_syn':
        gen_seeds = [12345, 15, 433, 57005, 7777]
        datasets = ['seed{}'.format(x) for x in gen_seeds]
    elif args.data == 'all':
        datasets = ['meps16', 'lsac', 'ACSP', 'credit', 'ACSE', 'ACSH', 'ACSI']
    elif args.data in all_supported_data:
        datasets = [args.data]
    else:
        raise ValueError(
            'The input "data" is not valid. CHOOSE FROM ["seed12345", "seed15", "seed433", "seed57005", "seed7777", "lsac", "cardio", "bank", "meps16", "credit", "ACSE", "ACSP", "ACSH", "ACSM", "ACSI"].')

    seeds = [1, 12345, 6, 2211, 15, 88, 121, 433, 500, 1121, 50, 583, 5278, 100000, 0xbeef, 0xcafe, 0xdead, 7777, 100, 923]

    models = ['lr', 'tr']

    targets = ['DI', 'FNR', 'FPR']

    if args.target == 'all':
        pass
    elif args.target in targets:
        targets = [args.target]
    else:
        raise ValueError('The input "target" is not valid. Choose from ["DI", "FNR", "FPR"].')

    if args.exec_n is None:
        raise ValueError(
            'The input "exec_n" is requried. Use "--exec_n 1" for a single execution.')
    elif type(args.exec_n) == str:
        raise ValueError(
            'The input "exec_n" requires integer. Use "--exec_n 1" for a single execution.')
    else:
        n_exec = int(args.exec_n)
        seeds = seeds[:n_exec]

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
        if args.weight == 'cap':
            models = ['tr']
    elif args.model in models:
        models = [args.model]
    else:
        raise ValueError('The input "model" is not valid. CHOOSE FROM ["lr", "tr"].')


    repo_dir = os.path.dirname(os.path.abspath(__file__))
    res_path = repo_dir + '/intermediate/models/'

    if args.run == 'parallel':
        tasks = []
        for data_name in datasets:
            for seed in seeds:
                for model_i in models:
                    for target in targets:
                        tasks.append([data_name, seed, model_i, args.weight, args.base, target, args.degree, res_path, args.opt])
        with Pool(cpu_count()//2) as pool:
            pool.starmap(run_retrain_model, tasks)
    else:
        raise ValueError('Do not support serial execution. Use "--run parallel"!')