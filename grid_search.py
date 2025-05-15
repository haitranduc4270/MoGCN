import pandas as pd
import itertools
import os
from tqdm import tqdm
import torch
import yaml
import argparse

from feature_selection.limma import limma_feature_selection
from feature_selection.chi_square import chi2_feature_selection
from feature_selection.lasso import lasso_feature_selection
from autoencoder.AE_run import run_ae


def run_grid_search(params):
    
    df_fpkm = pd.read_csv(params['fpkm_path'], index_col='Sample_ID')
    df_gistic = pd.read_csv(params['gistic_path'], index_col='Sample_ID')
    df_rppa = pd.read_csv(params['rppa_path'], index_col='Sample_ID')
    df_classes = pd.read_csv(params['sample_classes_path'], index_col='Sample_ID')
    
    exp_folder = os.path.join('experiments', params['exp_name'])
    os.makedirs(exp_folder, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    list_of_method = []
    if params['use_limma']:
        list_of_method.append(params['limma_thresholds'])
        list_of_method.append(params['chi2_thresholds'])
    else:
        list_of_method.append([-1])
        list_of_method.append([-1])
    
    if params['use_lasso']:
        list_of_method.append(params['lasso_alpha_values'])
    else:
        list_of_method.append([-1])
    
    if params['use_ae']:
        list_of_method.append(params['ae_latent_dims'])
    else:
        list_of_method.append([-1])
    
    combinations = list(itertools.product(*list_of_method))
    
    save_dict = {}  # Will store intermediate results for reuse
    
    for i, combination in tqdm(enumerate(combinations), total=len(combinations), desc="Grid Search Progress"):
        
        limma_threshold = float(combination[0])
        chi2_threshold = float(combination[1])
        lasso_alpha = combination[2]
        ae_latent_dim = combination[3]
        
        df_fpkm_copy = df_fpkm.copy()
        df_gistic_copy = df_gistic.copy()
        df_rppa_copy = df_rppa.copy()
        df_classes_copy = df_classes.copy()
        
        trial = f"exp_limma_{limma_threshold}_chi2_{chi2_threshold}_lasso_{lasso_alpha}_ae_{ae_latent_dim}.csv"
        trial_path = os.path.join(exp_folder, trial)
        
        # Feature selection - with caching
        if limma_threshold != -1:
            limma_key = f"limma_{limma_threshold}"
            chi2_key = f"chi2_{chi2_threshold}"
            
            # Check if we've already computed limma results for this threshold
            if limma_key in save_dict:
                df_fpkm_copy = save_dict[limma_key]['fpkm']
                df_rppa_copy = save_dict[limma_key]['rppa']
            else:
                df_fpkm_copy = limma_feature_selection(df_fpkm_copy, df_classes_copy, limma_threshold)
                df_rppa_copy = limma_feature_selection(df_rppa_copy, df_classes_copy, limma_threshold)
                save_dict[limma_key] = {
                    'fpkm': df_fpkm_copy,
                    'rppa': df_rppa_copy
                }
                
            # Check if we've already computed chi2 results for this threshold
            if chi2_key in save_dict:
                df_gistic_copy = save_dict[chi2_key]['gistic']
            else:
                df_gistic_copy = chi2_feature_selection(df_gistic_copy, df_classes_copy, chi2_threshold)
                save_dict[chi2_key] = {'gistic': df_gistic_copy}
        
        if lasso_alpha != -1:
            if limma_threshold != -1:
                lasso_key = f"lasso_{lasso_alpha}_limma_{limma_threshold}_chi2_{chi2_threshold}"
                if lasso_key in save_dict:
                    df_fpkm_copy = save_dict[lasso_key]['fpkm']
                    df_gistic_copy = save_dict[lasso_key]['gistic']
                    df_rppa_copy = save_dict[lasso_key]['rppa']
                else:
                    df_fpkm_copy = lasso_feature_selection(df_fpkm_copy, df_classes_copy, lasso_alpha)
                    df_gistic_copy = lasso_feature_selection(df_gistic_copy, df_classes_copy, lasso_alpha)
                    df_rppa_copy = lasso_feature_selection(df_rppa_copy, df_classes_copy, lasso_alpha)
                    save_dict[lasso_key] = {
                        'fpkm': df_fpkm_copy,
                        'gistic': df_gistic_copy,
                        'rppa': df_rppa_copy
                    }
            else:
                lasso_key = f"lasso_{lasso_alpha}"
                if lasso_key in save_dict:
                    df_fpkm_copy = save_dict[lasso_key]['fpkm']
                    df_gistic_copy = save_dict[lasso_key]['gistic']
                    df_rppa_copy = save_dict[lasso_key]['rppa']
                else:
                    df_fpkm_copy = lasso_feature_selection(df_fpkm_copy, df_classes_copy, lasso_alpha)
                    df_gistic_copy = lasso_feature_selection(df_gistic_copy, df_classes_copy, lasso_alpha)
                    df_rppa_copy = lasso_feature_selection(df_rppa_copy, df_classes_copy, lasso_alpha)
                    save_dict[lasso_key] = {
                        'fpkm': df_fpkm_copy,
                        'gistic': df_gistic_copy,
                        'rppa': df_rppa_copy
                    }
                                            
        if ae_latent_dim != -1:
            df_fpkm_copy = df_fpkm_copy.reset_index()
            df_gistic_copy = df_gistic_copy.reset_index()
            df_rppa_copy = df_rppa_copy.reset_index()
            df_latent = run_ae(df_fpkm_copy, df_gistic_copy, df_rppa_copy, latent_dim=ae_latent_dim, device=device)
        else:
            df_fpkm_copy = df_fpkm_copy.reset_index()
            df_gistic_copy = df_gistic_copy.reset_index()
            df_rppa_copy = df_rppa_copy.reset_index()
            df_latent = pd.merge(df_fpkm_copy, df_gistic_copy, on='Sample_ID', how='inner')
            df_latent = pd.merge(df_latent, df_rppa_copy, on='Sample_ID', how='inner')
            df_latent = df_latent.set_index('Sample_ID')
                        
        df_latent.to_csv(trial_path, index=True)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unified grid search for all feature selection methods')
    parser.add_argument('--exp_config', type=str, default='config/experiment_1.0.yaml')
    
    args = parser.parse_args()
    
    with open(args.exp_config, 'r') as f:
        params = yaml.safe_load(f)
    
    run_grid_search(params)