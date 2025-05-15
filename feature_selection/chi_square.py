from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
import pandas as pd

def chi2_feature_selection(df_data, df_classes, padj_threshold):
    """Perform chi-square feature selection with specified threshold"""
    labels = df_classes['class'].values
    pvals = {}
    
    for gene in df_data.columns:
        # Build contingency table
        try:
            table = pd.crosstab(df_data[gene], labels)
            if table.shape[0] > 1 and table.shape[1] > 1:
                _, p, _, _ = chi2_contingency(table)
                pvals[gene] = p
        except:
            continue
    
    if not pvals:
        print(f"No valid genes for chi-square test with threshold {padj_threshold}")
        return df_data.iloc[:, 0:min(10, df_data.shape[1])]
    
    # Multiple testing correction (FDR-BH)
    genes, raw_p = zip(*pvals.items())
    _, adj_p, _, _ = multipletests(raw_p, method="fdr_bh")
    chi2_df = pd.DataFrame({"pval": raw_p, "padj": adj_p}, index=genes)
    
    selected_chi2 = chi2_df[chi2_df["padj"] < padj_threshold].index.tolist()
    
    if len(selected_chi2) == 0:
        print(f"No significant genes found with threshold {padj_threshold}")
        return df_data.iloc[:, 0:min(10, df_data.shape[1])]
    
    return df_data[selected_chi2]