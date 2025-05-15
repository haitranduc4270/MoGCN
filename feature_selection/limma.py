import os
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from dotenv import load_dotenv

load_dotenv()

pandas2ri.activate()

# Import R packages
local_lib_path = os.getenv('LOCAL_R_PATH')
limma = importr('limma', lib_loc=local_lib_path)

def limma_feature_selection(df_expression, df_classes, adj_p_val_threshold):
    """Perform limma-based feature selection with specified threshold"""
    
    # Transpose the expression data for limma
    df_expression_t = df_expression.T
    
    # Convert to R objects
    r_expression = ro.conversion.py2rpy(df_expression_t)
    r_design = ro.FactorVector(df_classes['PAM50Call_RNAseq'].tolist())
    
    # Set in R global environment
    ro.globalenv['expression_data'] = r_expression
    ro.globalenv['design_factor'] = r_design
    
    # Create design matrix
    ro.r('design_matrix <- stats::model.matrix(~0 + design_factor)')
    ro.r('design_matrix <- as.data.frame(design_matrix)')
    ro.r('colnames(design_matrix) <- levels(design_factor)')
    
    # Fit linear model
    fit = limma.lmFit(ro.globalenv['expression_data'], ro.globalenv['design_matrix'])
    
    # Define contrasts
    contrasts_def = ['LumA - Basal', 'LumA - Her2', 'LumA - LumB', 
                     'LumB - Basal', 'LumB - Her2', 'Basal - Her2']
    significant_genes = set()
    
    for cont in contrasts_def:
        cont_matrix = limma.makeContrasts(cont, levels=ro.globalenv['design_matrix'].colnames)
        fit2 = limma.contrasts_fit(fit, cont_matrix)
        fit2 = limma.eBayes(fit2)
        top_table = limma.topTable(fit2, coef=1, number=ro.r('Inf'))
        top_df = pandas2ri.rpy2py_dataframe(top_table)
        sig_genes = top_df[top_df['adj.P.Val'] < adj_p_val_threshold].index
        significant_genes.update(sig_genes)
    
    # Select significant genes
    if len(significant_genes) == 0:
        print(f"No significant genes found with threshold {adj_p_val_threshold}")
        return df_expression.iloc[:, 0:min(10, df_expression.shape[1])]
    
    return df_expression[list(significant_genes)]