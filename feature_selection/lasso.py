from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def lasso_feature_selection(df_selected, df_classes, C_value):
    """Perform Lasso feature selection with specified C value"""
    # Prepare X (features) and y (labels)
    X = df_selected.values
    le = LabelEncoder()
    y = le.fit_transform(df_classes.loc[df_selected.index, 'PAM50Call_RNAseq'])
    
    # Fit logistic regression with L1 penalty (lasso)
    clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=2000, C=C_value, random_state=42)
    clf.fit(X, y)
    
    # Get selected features (where coefficients are non-zero)
    coefficients = clf.coef_  # Shape: (n_classes, n_features)
    selected = (coefficients != 0).any(axis=0)
    selected_features = df_selected.columns[selected]
    
    if len(selected_features) == 0:
        print(f"No features selected with C={C_value}")
        return df_selected.iloc[:, 0:min(10, df_selected.shape[1])]
    
    return df_selected[selected_features]