import logging
import sys
sys.path.append("F:\\guillermo\\Documentos\\Universidad\\Doctorado\\Articulos\\Fuzzy LORE\\Scamander")

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import time

from keras.models import load_model
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import pdist, squareform
import random

from sace.blackbox import BlackBox

from alibi.explainers import Counterfactual

from cf_eval.metrics import *

from experiments.config import *
from experiments.util import get_tabular_dataset

import hydra
from omegaconf import DictConfig, OmegaConf

# Watcher method


def experiment(cfe, bb, X_train, variable_features, metric, continuous_features, categorical_features_lists,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train, continuous_features_all,
               categorical_features_all, ratio_cont, nbr_features, filename_results, variable_features_flag,
               filename_cf, features_names):

    tol, lam_init, max_lam_steps, learning_rate_init = (0.04, 0.5, 35, 0.04)

    time_start = datetime.datetime.now()
    shape = (1,) + X_train.shape[1:]
    target_proba = 0.5
    target_class = 'other'
    max_iter = 1000

    predict_fn = lambda x: bb.predict_proba(x)

    time_train = (datetime.datetime.now() - time_start).total_seconds()

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)

    print(datetime.datetime.now(), dataset, black_box, cfe, metric, known_train, search_diversity)
    global_perf = 0
    for test_id, i in enumerate(index_test_instances):
        logging_message = f'Dataset: {dataset}, blackbox: {black_box}, progress: {(test_id + 1) / len(index_test_instances)}'
        logging.info(logging_message)
        try:
            x = X_test[i]
            y_val = bb.predict(x.reshape(1, -1))[0]
            x_eval_list = list()
            cf_list_all = list()

            time_start_i = datetime.datetime.now()
            perf = time.perf_counter()
            feature_range = (X_train.min(axis=0).reshape(shape),  # feature range for the perturbed instance
                                X_train.max(axis=0).reshape(shape))  # can be either a float or array of shape (1xfeatures)

            feature_range[0][:, variable_features] = x[variable_features]
            feature_range[1][:, variable_features] = x[variable_features]

            # initialize explainer
            exp = Counterfactual(predict_fn, shape=shape, target_proba=target_proba, tol=tol,
                                target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                                max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                                feature_range=feature_range)

            explanation = exp.explain(x.reshape(1, -1))
            
            if explanation.cf is not None:
                cf_list = explanation.cf['X']
                perf = time.perf_counter() - perf
                time_test = (datetime.datetime.now() - time_start_i).total_seconds()

                y_pred = bb.predict(X_test)
                impl = plausibility_fixed(x, bb, cf_list.reshape(1, -1), X_test, y_pred, continuous_features_all,
                                            categorical_features_all, X_train, ratio_cont)
                with open(filename_results, 'a+') as f:
                    f.write(f'{impl}\n')
            else:
                cf_list = np.array([])
        except:
            logging.warning(f'No counterfactual found for {dataset} - {black_box} - {i}')



def main(dataset, black_box, random_state):

    nbr_test = 100
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

    if dataset not in dataset_list:
        print('unknown dataset %s' % dataset)
        return -1

    if black_box not in blackbox_list:
        print('unknown black box %s' % black_box)
        return -1

    # if cfe not in cfe_list:
    #     print('unknown counterfactual explainer %s' % cfe)
    #     return -1

    print(datetime.datetime.now(), dataset, black_box)

    data = get_tabular_dataset(dataset, path_dataset, normalize=normalize, test_size=test_size,
                               random_state=random_state, encode=None if black_box == 'LGBM' else 'onehot')
    X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

    features_names = data['feature_names']
    variable_features = data['variable_features']
    continuous_features = data['continuous_features']
    continuous_features_all = data['continuous_features_all']
    categorical_features_lists = data['categorical_features_lists']
    categorical_features_all = data['categorical_features_all']
    nbr_features = data['n_cols']
    ratio_cont = data['n_cont_cols'] / nbr_features


    import tensorflow as tf
    tf.compat.v1.disable_eager_execution()

    if black_box in ['DT', 'RF', 'SVM', 'NN', 'LGBM']:
        bb = pickle.load(open(path_models + '%s_%s.pickle' % (dataset, black_box), 'rb'))
        # if black_box == 'RF':
        #     bb.n_jobs = 5
    elif black_box in ['DNN']:

        bb = load_model(path_models + '%s_%s.h5' % (dataset, black_box))
    else:
        print('unknown black box %s' % black_box)
        raise Exception

    bb = BlackBox(bb)

    filename_results = path_results + 'impl_%s_%s_cfw.csv' % (dataset, black_box)
    filename_cf = path_cf + 'cf_%s_%s_cfw.csv' % (dataset, black_box)

    experiment('cfw', bb, X_train, variable_features, metric,
               continuous_features, categorical_features_lists,
               X_test, nbr_test, search_diversity, dataset, black_box, known_train,
               continuous_features_all, categorical_features_all, ratio_cont, nbr_features,
               filename_results, variable_features_flag, filename_cf, features_names)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    main(cfg.dataset.name, cfg.bb.name, cfg.rs.value)


if __name__ == "__main__":
    my_app()

