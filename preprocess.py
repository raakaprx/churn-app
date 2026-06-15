import pandas as pd
import numpy as np

def test_prep(test_df):

    test_df['TotalCharges'] = pd.to_numeric(
        test_df['TotalCharges'],
        errors='coerce'
    )

    test_df.replace(
        ['No internet service', 'No phone service'],
        'No',
        inplace=True
    )

    cat_cols = [
        'gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'PaperlessBilling',
        'PaymentMethod'
    ]

    test_df = pd.concat(
        [test_df, pd.get_dummies(test_df[cat_cols])],
        axis='columns'
    )

    test_df = test_df.drop(columns=cat_cols)

    test_df['Churn'] = np.where(
        test_df['Churn'] == 'Yes',
        1,
        0
    )

    condition = [
        ((test_df.tenure >= 0) & (test_df.tenure <= 12)),
        ((test_df.tenure > 12) & (test_df.tenure <= 24)),
        ((test_df.tenure > 24) & (test_df.tenure <= 36)),
        ((test_df.tenure > 36) & (test_df.tenure <= 48)),
        ((test_df.tenure > 48) & (test_df.tenure <= 60)),
        ((test_df.tenure > 60))
    ]

    choice = [0,1,2,3,4,5]

    test_df['tenure_range'] = np.select(
        condition,
        choice
    )

    test_df['MonthlyCharges'] = np.log1p(
        test_df['MonthlyCharges']
    )

    test_df['TotalCharges'] = np.log1p(
        test_df['TotalCharges']
    )

    return test_df