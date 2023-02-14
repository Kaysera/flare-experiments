# %%
import sys
import os
sys.path.append("F:\\guillermo\\Documentos\\Universidad\\Doctorado\\Articulos\\Fuzzy LORE\\Scamander")

import pickle
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import train_test_split
from scipy.stats import median_abs_deviation

import time
import random
import logging

# %%
from sace.blackbox import BlackBox
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
           'plausibility_nbr_actionable_cf', 'plausibility_nbr_valid_actionable_cf', 'plausibility_fixed'
]

CONFIGS = {('SVM', 'adult'): 'md3_ns500_me5',
 ('SVM', 'compas'): 'md3_ns500_me3',
 ('SVM', 'german'): 'md3_ns500_me3',
 ('SVM', 'fico'): 'md4_ns2000_me5',
 ('NN', 'adult'): 'md3_ns500_me1',
 ('NN', 'compas'): 'md-1_ns500_me1',
 ('NN', 'german'): 'md3_ns500_me10',
 ('NN', 'fico'): 'md4_ns1500_me10'}

def process_config(config):
    md = int(config.split('_')[0][2:])
    ns = int(config.split('_')[1][2:])
    me = int(config.split('_')[2][2:])
    return md, ns, me

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
def build_explainer(instance, target, mad, max_depth, size, class_name, blackbox, dataset, X_train, get_division, df_numerical_columns, df_categorical_columns, f_method, cf_method, cont_idx, disc_idx, cf_dist='moth', min_num_examples=10, neighrange='std', threshold=None, fuzzy_threshold=0.0001):
    neighborhood = SamplingNeighborhood(instance, size, class_name, blackbox, dataset, np.row_stack([X_train, instance]), len(X_train), neighbor_generation='fast', neighbor_range=neighrange)
    neighborhood.fit()
    neighborhood.fuzzify(get_division,
                            class_name=class_name,
                            df_numerical_columns=df_numerical_columns,
                            df_categorical_columns=df_categorical_columns,
                            th=threshold)
    decoded_instance = decode_instance(instance, dataset, neighborhood.get_fuzzy_variables())

    explainer = FDTExplainer()
    if max_depth == -1:
        max_depth = len(X_train[0])
    explainer.fit(decoded_instance.reshape(1, -1), target, neighborhood, df_numerical_columns, f_method, cf_method, max_depth=max_depth, min_num_examples=min_num_examples, fuzzy_threshold=fuzzy_threshold, cont_idx=cont_idx, disc_idx=disc_idx, mad=mad, cf_dist=cf_dist)
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


def main(ds, black_box, random_state, max_depth, size, cf_dist, min_num_examples, idx_record2explain, neighrange):
    # %%
    random.seed(random_state)
    np.random.seed(random_state)

    # %%
    # ds = 'german'
    # black_box = 'NN'

    nbr_test = 100
    n_path_models = './models/'
    bb = pickle.load(open(n_path_models + '%s_%s.pickle' % (ds, black_box), 'rb'))

    max_depth, size, min_num_examples = process_config(CONFIGS[(black_box, ds)])
    # %%
    dataset = DATASETS[ds](normalize=True)

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    index_test_instances = np.random.choice(range(len(X_test)), nbr_test)
    class_name = dataset['class_name']
    get_division = 'entropy'

    df_numerical_columns = [col for col in dataset['continuous'] if col != class_name]
    df_categorical_columns = [col for col in dataset['discrete'] if col != class_name]
    features = [col for col in dataset['columns'] if col != class_name]

    cont_idx = [key for key, val in dataset['idx_features'].items() if val in df_numerical_columns]
    disc_idx = [key for key, val in dataset['idx_features'].items() if val in df_categorical_columns]

    features_idx = {val: key for key, val in dataset['idx_features'].items()}
    f_method = 'mr_factual'
    cf_method = 'd_counterfactual'
    path_results = path + 'experimentos_vecindariosos_v2\\'
    filename_results = path_results + 'cf_performance_%s_%s_%s_%s_flore.csv' % (ds, black_box, idx_record2explain, neighrange)
    # %%
    logging_message = f'Dataset: {ds}, blackbox: {black_box}, max_depth: {max_depth}, size: {size}, min_num_examples: {min_num_examples}, exp_id: {idx_record2explain}'
    logging.info(logging_message)
    x_eval_list = list()
    instance = X_test[idx_record2explain]
    target = bb.predict(instance.reshape(1, -1))
    mad = get_dataset_mad(X_train, cont_idx)
    perf = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        explainer, neighborhood, decoded_instance = build_explainer(instance, target, mad, max_depth, size, class_name, bb, dataset, X_train, get_division, df_numerical_columns, df_categorical_columns, f_method, cf_method, cont_idx, disc_idx, cf_dist, min_num_examples, neighrange, fuzzy_threshold=0.0001)
    

    # %%
    factual, counterfactual = explainer.explain()
    decoded_cf_instance = apply_counterfactual(decoded_instance, counterfactual, features_idx, dataset['features_type'])
    cf_instance = apply_counterfactual(instance, counterfactual, features_idx, dataset['features_type'], dataset['label_encoder'])
    perf = time.perf_counter() - perf

    print(cf_instance)
    # %%
    # x_eval = evaluate_cf_list(np.array([cf_instance]), 
    #                             instance, 
    #                             bb, 
    #                             target, 
    #                             1, 
    #                             list(range(len(features))),
    #                             cont_idx, 
    #                             disc_idx, 
    #                             X_train, 
    #                             X_test,
    #                             len(cont_idx) / len(features),
    #                             len(features))
    
    # x_eval['dataset'] = ds
    # x_eval['black_box'] = black_box
    # x_eval['method'] = 'flore'
    # x_eval['idx'] = idx_record2explain
    # x_eval['perf'] = perf
    # x_eval_list.append(x_eval)
    # df = pd.DataFrame(data=x_eval_list)
    # df = df[columns + ['perf']]
    
    # if not os.path.isfile(filename_results):
    #     df.to_csv(filename_results, index=False)
    # else:
    #     df.to_csv(filename_results, mode='a', index=False, header=False)

# %%

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def my_app(cfg : DictConfig) -> None:
    try:
        main(
            cfg.dataset.name, 
            cfg.bb.name, 
            cfg.rs.value,
            cfg.maxdepth.value,
            cfg.nsize.value,
            cfg.cfdist.name,
            cfg.minexamples.value,
            cfg.instance.value,
            cfg.neighrange
        )
    except:
        logging.error('No counterfactual found')



if __name__ == "__main__":
    # my_app()
    main(
        'adult', 
        'NN', 
        42,
        -1,
        500,
        'moth',
        1,
        7270,
        0.01
    )