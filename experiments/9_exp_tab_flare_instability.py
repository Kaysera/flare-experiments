# %%
import sys
import os
sys.path.append("F:\\guillermo\\Documentos\\Universidad\\Doctorado\\Articulos\\Fuzzy LORE\\Scamander")
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from scipy.stats import median_abs_deviation

import time
import random
import logging

# %%
from sace.blackbox import BlackBox
from sace.random_sace import RandomSACE

from cf_eval.metrics import *

from experiments.config import *

# %%
from teacher.explanation import FDTExplainer
from teacher.neighbors import SamplingNeighborhood
from teacher.datasets import load_compas, load_german, load_adult, load_heloc

import hydra
from omegaconf import DictConfig, OmegaConf

# %%
DATASETS = {
    'adult': load_adult,
    'compas': load_compas,
    'fico': load_heloc,
    'german': load_german
}

columns = ['dataset',  'black_box', 'method', 'idx',
           'nbr_cf', 'nbr_valid_cf', 'perc_valid_cf', 'perc_valid_cf_all', 'nbr_actionable_cf', 'perc_actionable_cf',
           'perc_actionable_cf_all', 'nbr_valid_actionable_cf', 'perc_valid_actionable_cf',
           'perc_valid_actionable_cf_all', 'avg_nbr_violations_per_cf', 'avg_nbr_violations',
           'distance_l2', 'distance_mad', 'distance_j', 'distance_h', 'distance_l2j', 'distance_mh',
           'distance_l2_min', 'distance_mad_min', 'distance_j_min', 'distance_h_min', 'distance_l2j_min',
           'distance_mh_min', 'distance_l2_max', 'distance_mad_max', 'distance_j_max', 'distance_h_max',
           'distance_l2j_max', 'distance_mh_max', 'avg_nbr_changes_per_cf', 'avg_nbr_changes', 'diversity_l2',
           'diversity_mad', 'diversity_j', 'diversity_h', 'diversity_l2j', 'diversity_mh', 'diversity_l2_min',
           'diversity_mad_min', 'diversity_j_min', 'diversity_h_min', 'diversity_l2j_min', 'diversity_mh_min',
           'diversity_l2_max', 'diversity_mad_max', 'diversity_j_max', 'diversity_h_max', 'diversity_l2j_max',
           'diversity_mh_max', 'count_diversity_cont', 'count_diversity_cate', 'count_diversity_all',
           'accuracy_knn_sklearn', 'accuracy_knn_dist', 'lof', 'delta', 'delta_min', 'delta_max',
           'plausibility_sum', 'plausibility_max_nbr_cf', 'plausibility_nbr_cf', 'plausibility_nbr_valid_cf',
           'plausibility_nbr_actionable_cf', 'plausibility_nbr_valid_actionable_cf'
]

# %%
def get_dataset_mad(X, cont_idx):
    mad = {}
    for i in cont_idx:
        mad[i] = median_abs_deviation(X[:, i])
        if mad[i] == 0: # ÑAPA TEMPORAL, NO SE SI ESTO ESTA BIEN O QUE
            mad[i] += 1
    return mad

# %%
def decode_instance(instance, dataset, fuzzy_variables):
    decoded_instance = []
    for i, var in enumerate(fuzzy_variables):
        try:
            decoded_instance.append(dataset['label_encoder'][var.name].inverse_transform(np.array([instance[i]], dtype=int))[0])
        except:
            decoded_instance += [instance[i]]
    return np.array(decoded_instance, dtype='object')

# %%
def build_explainer(instance, target, mad, max_depth, size, class_name, blackbox, dataset, X_train, get_division, df_numerical_columns, df_categorical_columns, f_method, cf_method, cont_idx, disc_idx, min_num_examples=10, fuzzy_threshold=0.0001):
    neighborhood = SamplingNeighborhood(instance, size, class_name, blackbox, dataset, np.row_stack([X_train, instance]), len(X_train), neighbor_generation='fast')
    neighborhood.fit()
    neighborhood.fuzzify(get_division,
                            class_name=class_name,
                            df_numerical_columns=df_numerical_columns,
                            df_categorical_columns=df_categorical_columns)
    decoded_instance = decode_instance(instance, dataset, neighborhood.get_fuzzy_variables())

    explainer = FDTExplainer()
    explainer.fit(decoded_instance.reshape(1, -1), target, neighborhood, df_numerical_columns, f_method, cf_method, max_depth=max_depth, min_num_examples=min_num_examples, fuzzy_threshold=fuzzy_threshold, cont_idx=cont_idx, disc_idx=disc_idx, mad=mad)
    return explainer, neighborhood, decoded_instance

# %%
def apply_counterfactual(instance, counterfactual, features_idx, features_type, label_encoder=None):
    TYPES = {
        'integer': float,
        'string': str,
        'float': float
    }
    cf_instance = instance.copy()
    for var, val in counterfactual:
        if label_encoder and var in label_encoder:
            cf_instance[features_idx[var]] = label_encoder[var].transform([TYPES[features_type[var]](val)])
        else:
            cf_instance[features_idx[var]] = TYPES[features_type[var]](val)
    
    return cf_instance


def main(ds, black_box, random_state):

    # %%
    random.seed(random_state)
    np.random.seed(random_state)

    # %%
    # ds = 'german'
    # black_box = 'NN'

    nbr_test = 2
    nbr_exp = nbr_test
    n_path_models = './models/'
    bb = pickle.load(open(n_path_models + '%s_%s.pickle' % (ds, black_box), 'rb'))
    bb = BlackBox(bb)


    # %%
    dataset = DATASETS[ds](normalize=True)

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)
    size = 1000
    class_name = dataset['class_name']
    get_division = 'entropy'

    df_numerical_columns = [col for col in dataset['continuous'] if col != class_name]
    df_categorical_columns = [col for col in dataset['discrete'] if col != class_name]
    features = [col for col in dataset['columns'] if col != class_name]

    cont_idx = [key for key, val in dataset['idx_features'].items() if val in df_numerical_columns]
    disc_idx = [key for key, val in dataset['idx_features'].items() if val in df_categorical_columns]

    features_idx = {val: key for key, val in dataset['idx_features'].items()}
    max_depth = np.nan
    f_method = 'mr_factual'
    cf_method = 'd_counterfactual'

    metric_calc_dist = ('euclidean', 'jaccard')

    exp_calc_dist = RandomSACE(list(range(len(features))), weights=None, metric=metric_calc_dist, feature_names=None,
                               continuous_features=cont_idx,
                               categorical_features_lists=disc_idx,
                               normalize='standard', pooler=None, n_attempts=100, n_max_attempts=1000, proba=0.5)
    exp_calc_dist.fit(bb, X_train)

    y_pred = bb.predict(X_test)
    class_values = sorted(np.unique(y_pred))
    nbr_exp_per_class = nbr_exp // len(class_values)

    couples_to_test = list()
    for class_val in class_values:
        X_test_y = X_test[y_pred == class_val]
        for i, x in enumerate(X_test_y):
            neigh_dist = exp_calc_dist.cdist(x.reshape(1, -1), X_test_y)
            idx_neigh = np.argsort(neigh_dist, kind='stable')[0]
            closest_idx = idx_neigh[0] if i != idx_neigh[0] else idx_neigh[1]
            couples_to_test.append((x, X_test_y[closest_idx]))
            if i >= nbr_exp_per_class:
                break

    filename_stability = path_results + 'instability_%s_%s_flore.csv' % (ds, black_box)
    print(filename_stability)

    # %%
    for test_id, couple in enumerate(couples_to_test):
        logging_message = f'Dataset: {dataset}, blackbox: {black_box}, progress: {(test_id + 1) / len(couples_to_test)}'
        logging.info(logging_message)
        x1 = couple[0]
        x2 = couple[1]        
        x_eval_list = list()
        
        instance = x1
        target = bb.predict(instance.reshape(1, -1))
        mad = get_dataset_mad(X_train, cont_idx)
        perf = time.perf_counter()
        explainer, neighborhood, decoded_instance = build_explainer(instance, target, mad, max_depth, size, class_name, bb, dataset, X_train, get_division, df_numerical_columns, df_categorical_columns, f_method, cf_method, cont_idx, disc_idx, min_num_examples=10, fuzzy_threshold=0.0001)

        # %%
        factual, counterfactual = explainer.explain()
        decoded_cf_instance = apply_counterfactual(decoded_instance, counterfactual, features_idx, dataset['features_type'])
        cf_list1 = np.array([apply_counterfactual(instance, counterfactual, features_idx, dataset['features_type'], dataset['label_encoder'])])


        instance = x2
        target = bb.predict(instance.reshape(1, -1))
        mad = get_dataset_mad(X_train, cont_idx)
        perf = time.perf_counter()
        explainer, neighborhood, decoded_instance = build_explainer(instance, target, mad, max_depth, size, class_name, bb, dataset, X_train, get_division, df_numerical_columns, df_categorical_columns, f_method, cf_method, cont_idx, disc_idx, min_num_examples=10, fuzzy_threshold=0.0001)

        # %%
        factual, counterfactual = explainer.explain()
        decoded_cf_instance = apply_counterfactual(decoded_instance, counterfactual, features_idx, dataset['features_type'])
        cf_list2 = np.array([apply_counterfactual(instance, counterfactual, features_idx, dataset['features_type'], dataset['label_encoder'])])
        # %%
        d_x1x2 = exp_calc_dist.cdist(x1.reshape(1, -1), x2.reshape(1, -1))[0][0]

        sum_c1c2 = 0.0
        for c1 in cf_list1:
            for c2 in cf_list2:
                d_c1c2 = exp_calc_dist.cdist(c1.reshape(1, -1), c2.reshape(1, -1))[0][0]
                sum_c1c2 += d_c1c2

        if len(cf_list1) > 0 and len(cf_list2) > 0:
            inst_x1x2 = 1.0 / (1.0 + d_x1x2) * 1.0 / (len(cf_list1) * len(cf_list2)) * sum_c1c2
        else:
            inst_x1x2 = np.nan

        x_eval = dict()
        x_eval['inst_x1x2'] = inst_x1x2

        x_eval['dataset'] = dataset
        x_eval['black_box'] = black_box
        x_eval['method'] = 'flore'
        x_eval['couple_idx'] = test_id
        x_eval_list.append(x_eval)

        df = pd.DataFrame(data=x_eval_list)
        df = df[['dataset', 'black_box', 'method', 'couple_idx', 'inst_x1x2']]

        if not os.path.isfile(filename_stability):
            df.to_csv(filename_stability, index=False)
        else:
            df.to_csv(filename_stability, mode='a', index=False, header=False)

# %%

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    main(cfg.dataset.name, cfg.bb.name, cfg.rs.value)


if __name__ == "__main__":
    main('adult', 'NN', 42)