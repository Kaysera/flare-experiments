import sys
sys.path.append("F:\\guillermo\\Documentos\\Universidad\\Doctorado\\Articulos\\Fuzzy LORE\\Scamander")


import os
from joblib import Parallel, delayed

import pickle
import shutil
import datetime
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
import random

from keras.models import load_model

from sace.blackbox import BlackBox

from alibi.explainers import CounterfactualProto

from cf_eval.metrics import *

from experiments.config import *
from experiments.util import get_tabular_dataset

import hydra
from omegaconf import DictConfig, OmegaConf


def experiment(cfe, filename, i, bb, X_train, variable_features, metric, continuous_features, categorical_features_lists,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train, continuous_features_all,
               categorical_features_all, ratio_cont, nbr_features, filename_results, variable_features_flag,
               filename_cf, features_names):
    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()


    conf = {
        'beta': np.random.randint(1, 40) * 0.01,
        'c_init': np.random.randint(1, 10),
        'c_steps': np.random.randint(10, 50),
    }

    with open(os.path.join('retest_cegp', filename), 'r') as f:
        cf_ratio, conf['beta'], conf['c_init'], conf['c_steps'] = f.read().split(',')


    shape = (1,) + X_train.shape[1:]  # instance shape
    max_iterations = 500
    total_cf = 0
    beta = float(conf['beta'])  # weight of the L1 loss term
    c_init = int(conf['c_init'])  # initial weight c of the loss term encouraging to predict a different class (PN) or
    c_steps = int(conf['c_steps'])  # nb of updates for c
    max_iterations = 500  # nb of iterations per value of c

    predict_fn = lambda x: bb.predict_proba(x)  # only pass the predict fn which takes numpy arrays to CEM

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    for test_id, i in enumerate(index_test_instances):
        logging_message = f'Dataset: {dataset}, blackbox: {black_box}, exp_id: {i}, progress: {(test_id + 1) / len(index_test_instances)}'
        logging.info(logging_message)
        
        try:
            x = X_test[i]
            x_eval_list = list()

            y_val = bb.predict(x.reshape(1, -1))[0]

            feature_range = (X_train.min(axis=0).reshape(shape),  # feature range for the perturbed instance
                            X_train.max(axis=0).reshape(shape))  # can be either a float or array of shape (1xfeatures)
            feature_range[0][:, variable_features] = x[variable_features]
            feature_range[1][:, variable_features] = x[variable_features]

            exp = CounterfactualProto(predict_fn,
                                    shape,
                                    beta=beta,
                                    # cat_vars=cat_vars_ohe,
                                    # ohe=True,  # OHE flag
                                    max_iterations=max_iterations,
                                    feature_range=feature_range,
                                    c_init=c_init,
                                    c_steps=c_steps
                                    )

            exp.fit(X_train, d_type='abdm', disc_perc=[25, 50, 75])

            explanation = exp.explain(x.reshape(1, -1))

            if explanation.cf is not None:
                cf_list = explanation.cf['X']
                total_cf += 1
                logging.info('Counterfactual found')

            else:
                logging.info('No counterfactual found')
                cf_list = np.array([])

            k = 1
            x_eval = evaluate_cf_list(cf_list, x, bb, y_val, k, variable_features,
                                        continuous_features_all, categorical_features_all, X_train, X_test,
                                        ratio_cont, nbr_features)

            x_eval['dataset'] = dataset
            x_eval['black_box'] = black_box
            x_eval['method'] = cfe
            x_eval['idx'] = i
            x_eval['k'] = k
            x_eval['known_train'] = known_train
            x_eval['search_diversity'] = search_diversity
            x_eval['time_train'] = 0
            x_eval['time_test'] = 0
            x_eval['runtime'] = 0
            x_eval['metric'] = metric if isinstance(metric, str) else '.'.join(metric)
            x_eval['variable_features_flag'] = variable_features_flag
            x_eval['filename'] = filename
            x_eval['instability_si'] = 0

            x_eval_list.append(x_eval)

            df = pd.DataFrame(data=x_eval_list)
            df = df[columns + ['filename']]

            if not os.path.isfile(filename_results):
                df.to_csv(filename_results, index=False)
            else:
                df.to_csv(filename_results, mode='a', index=False, header=False)


        except Exception as e:
            logging_message = f'Dataset: {dataset}, blackbox: {black_box}, exp_id: {i}, instance: {test_id}'
            logging.error(e)

    return total_cf

def parse_file_name(file_name):
    db, bb, idx, algo = file_name[:-4].split('_')[1:]
    return db, bb, idx, algo

def main(filename, random_state):
    dataset, black_box, idx, algo = parse_file_name(filename)

    logging_message = f'[GENERAL] Launching experiment :{black_box}-{dataset}-{idx}-{algo}'
    logging.info(logging_message)
    logging.info(filename)
    nbr_test = 10
    normalize = 'standard'

    # dataset = sys.argv[1]
    # black_box = sys.argv[2]
    # cfe = sys.argv[3]
    # nbr_test = 100 if len(sys.argv) < 5 else int(sys.argv[4])
    known_train = True
    search_diversity = False
    metric = 'none'
    variable_features_flag = True
    random.seed(random_state)
    np.random.seed(random_state)

    data = get_tabular_dataset(dataset, path_dataset, normalize=normalize, test_size=test_size,
                               random_state=random_state, encode=None if black_box == 'LGBM' else 'onehot')
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']
    class_values = data['class_values']
    if dataset == 'titanic':
        class_values = ['Not Survived', 'Survived']
    features_names = data['feature_names']
    variable_features = data['variable_features']
    variable_features_names = data['variable_features_names']
    continuous_features = data['continuous_features']
    continuous_features_all = data['continuous_features_all']
    categorical_features_lists = data['categorical_features_lists']
    categorical_features_lists_all = data['categorical_features_lists_all']
    categorical_features_all = data['categorical_features_all']
    continuous_features_names = data['continuous_features_names']
    categorical_features_names = data['categorical_features_names']
    scaler = data['scaler']
    nbr_features = data['n_cols']
    ratio_cont = data['n_cont_cols'] / nbr_features

    shutil.copy(path_models + '%s_%s.pickle' % (dataset, black_box), path_models + 'tmp\\%s_%s_%s_%s.pickle'% (dataset, black_box, idx, algo))

    if black_box in ['DT', 'RF', 'SVM', 'NN', 'LGBM']:
        bb = pickle.load(open(path_models + 'tmp\\%s_%s_%s_%s.pickle'% (dataset, black_box, idx, algo), 'rb'))
        # if black_box == 'RF':
        #     bb.n_jobs = 5
    elif black_box in ['DNN']:

        bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    else:
        print('unknown black box %s' % black_box)
        raise Exception

    bb = BlackBox(bb)

    filename_results = path_results + 'paramsearch_%s_%s_cegp.csv' % (dataset, black_box)
    filename_cf = path_cf + 'cf_%s_%s_cegp.csv' % (dataset, black_box)

    experiment('cegp', filename, 0, bb, X_train, variable_features, metric,
               continuous_features, categorical_features_lists,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train,
               continuous_features_all, categorical_features_all, ratio_cont, nbr_features,
               filename_results, variable_features_flag, filename_cf, features_names)

    os.remove(path_models + 'tmp\\%s_%s_%s_%s.pickle'% (dataset, black_box, idx, algo))

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    main(cfg.filename.value, cfg.rs.value)


if __name__ == "__main__":
    my_app()
    # main(
    #     'paramsearch_compas_NN_0_cegp.csv',
    #     42
    # )



