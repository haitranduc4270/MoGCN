import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
from scipy import stats
import seaborn as sns

predict = pd.read_csv('result/GAT_predicted_data.csv');
# clinical = pd.read_csv('data/clinical/TCGA-BRCA.csv');

# clinical['id_key'] = clinical.iloc[:, 0].str.extract(r'([^-]+-[^-]+)$')
predict = predict.rename(columns={predict.columns[0]: 'id_key'})

clinical_data = pd.read_csv('data/clinical/Human__TCGA_BRCA__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi', sep='\t')
clinical_data = clinical_data.set_index('attrib_name')
clinical_data = clinical_data.transpose()
clinical_data = clinical_data.reset_index().rename(columns={'index': 'patient_id'})

clinical_data['id_key'] = clinical_data['patient_id'].str.replace('.', '-', regex=False).replace('^TCGA\\-', '', regex=True)

# # Hiển thị dữ liệu

result = pd.merge(predict, clinical_data, on='id_key', how='left')
# result = pd.merge(predict, clinical, on='id_key', how='left')

print(result)

# Survival Analysis
df = result.copy()
df['survival_time'] = df['exp.brca_selected.days_to_death'].fillna(df['exp.brca_selected.days_to_last_follow_up'])
df['event'] = df['exp.brca_selected.paper_vital_status'].apply(lambda x: 1 if str(x).lower() == 'dead' else 0)

# Remove rows with missing survival data
df = df.dropna(subset=['survival_time', 'event', 'predict_label'])

# Plot Kaplan-Meier curves
plt.figure(figsize=(10, 6))
kmf = KaplanMeierFitter()

for label in sorted(df['predict_label'].unique()):
    subset = df[df['predict_label'] == label]
    kmf.fit(subset['survival_time'], subset['event'], label=f'Group {label}')
    kmf.plot_survival_function(ci_show=False)

plt.title("Kaplan-Meier Survival Curve by Predicted Label")
plt.xlabel("Days")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.tight_layout()
plt.savefig('survival_curves.png')
plt.close()

# Log-rank test if there are 2 groups
if df['predict_label'].nunique() == 2:
    g1, g2 = df['predict_label'].unique()
    group1 = df[df['predict_label'] == g1]
    group2 = df[df['predict_label'] == g2]
    results = logrank_test(group1['survival_time'], group2['survival_time'],
                          event_observed_A=group1['event'],
                          event_observed_B=group2['event'])
    print(f"Log-rank test p-value: {results.p_value}")

# Clinical Parameter Analysis
# Convert categorical variables to numeric where needed
result['years_to_birth'] = pd.to_numeric(result['years_to_birth'], errors='coerce')
result['number_of_lymph_nodes'] = pd.to_numeric(result['number_of_lymph_nodes'], errors='coerce')

# Function to calculate p-values for different types of variables
def calculate_p_value(subtype, clinical_var):
    if clinical_var in ['years_to_birth', 'number_of_lymph_nodes']:
        # For continuous variables, use ANOVA
        groups = [group for _, group in result.groupby('predict_label')[clinical_var]]
        return stats.f_oneway(*groups)[1]
    else:
        # For categorical variables, use Chi-square test
        contingency = pd.crosstab(result['predict_label'], result[clinical_var])
        return stats.chi2_contingency(contingency)[1]

# List of clinical parameters to analyze
clinical_params = [
    'years_to_birth',  # age
    'gender',
    'pathologic_stage',
    'pathology_T_stage',  # tumor size
    'pathology_N_stage',  # lymph node status
    'pathology_M_stage'   # metastasis status
]

# Calculate p-values and -log10(p-values)
p_values = {}
log_p_values = {}
significant_params = []

for param in clinical_params:
    p_val = calculate_p_value('predict_label', param)
    p_values[param] = p_val
    log_p_values[param] = -np.log10(p_val)
    if p_val < 0.05:
        significant_params.append(param)

# Plot -log10 P-values
plt.figure(figsize=(10, 6))
plt.bar(log_p_values.keys(), log_p_values.values())
plt.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
plt.xticks(rotation=45)
plt.ylabel('-log10(P-value)')
plt.title('Statistical Significance of Clinical Parameters')
plt.legend()
plt.tight_layout()
plt.savefig('clinical_significance.png')
plt.close()

# Print results
print("\nNumber of significant clinical parameters:", len(significant_params))
print("\nSignificant parameters:", significant_params)
print("\nP-values for each parameter:")
for param, p_val in p_values.items():
    print(f"{param}: {p_val:.2e}")

# Analyze relationships between subtypes and clinical features
print("\nSubtype Analysis:")
for param in clinical_params:
    if param in ['years_to_birth', 'number_of_lymph_nodes']:
        print(f"\n{param} by subtype:")
        print(result.groupby('predict_label')[param].describe())
    else:
        print(f"\n{param} distribution by subtype:")
        print(pd.crosstab(result['predict_label'], result[param]))

# Save results to CSV
df.to_csv('survival_analysis_results.csv', index=False)
