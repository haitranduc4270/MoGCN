import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

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

# df = result.copy()
# df['survival_time'] = df['exp.brca_selected.days_to_death'].fillna(df['exp.brca_selected.days_to_last_follow_up'])
# df['event'] = df['exp.brca_selected.paper_vital_status'].apply(lambda x: 1 if str(x).lower() == 'dead' else 0)

# # Xóa những dòng thiếu thông tin cần thiết
# df = df.dropna(subset=['survival_time', 'event', 'predict_label'])

# # Bước 2: Vẽ Kaplan-Meier theo nhóm predict_label
# kmf = KaplanMeierFitter()
# plt.figure(figsize=(10, 6))

# for label in sorted(df['predict_label'].unique()):
#     subset = df[df['predict_label'] == label]
#     kmf.fit(subset['survival_time'], subset['event'], label=f'Group {label}')
#     kmf.plot_survival_function(ci_show=False)

# plt.title("Kaplan-Meier Survival Curve by Predicted Label")
# plt.xlabel("Days")
# plt.ylabel("Survival Probability")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Bước 3: Log-rank test nếu có 2 nhóm
# if df['predict_label'].nunique() == 2:
#     g1, g2 = df['predict_label'].unique()
#     group1 = df[df['predict_label'] == g1]
#     group2 = df[df['predict_label'] == g2]
#     results = logrank_test(group1['survival_time'], group2['survival_time'],
#                            event_observed_A=group1['event'],
#                            event_observed_B=group2['event'])
#     print(f"Log-rank test p-value: {results.p_value}")
