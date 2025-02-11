import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf

def perform_ttest(df, group_col, value_col, group1, group2):
    """
    Performs a Student's t-test to compare means of two groups.

    Args:
        df (pandas.DataFrame): DataFrame.
        group_col (str): Name of the column defining groups.
        value_col (str): Name of the column with values to compare.
        group1 (str): Value of group 1 in 'group_col'.
        group2 (str): Value of group 2 in 'group_col'.

    Returns:
        tuple: t-statistic and p-value.
    """
    group1_data = df[df[group_col] == group1][value_col].dropna()
    group2_data = df[df[group_col] == group2][value_col].dropna()
    if not group1_data.empty and not group2_data.empty:
        t_statistic, p_value = stats.ttest_ind(group1_data, group2_data)
        return t_statistic, p_value
    else:
        return None, None

def perform_anova(df, group_col, value_col):
    """
    Performs ANOVA to compare means across multiple groups and returns the ANOVA table.

    Args:
        df (pandas.DataFrame): DataFrame.
        group_col (str): Name of the column defining groups.
        value_col (str): Name of the column with values to compare.

    Returns:
        pandas.DataFrame or tuple: ANOVA table as DataFrame if successful, otherwise None, None for F-statistic and p-value.
    """
    formula = f'{value_col} ~ C({group_col})' # Define formula for ANOVA
    try:
        model = smf.ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model) # Get ANOVA table
        return anova_table # Return ANOVA table as DataFrame
    except Exception as e: # Catch potential errors like not enough groups, etc.
        print(f"Error performing ANOVA: {e}")
        return None # Return None if ANOVA cannot be performed

def perform_linear_regression(df, y_col, x_cols):
    """
    Performs multiple linear regression.

    Args:
        df (pandas.DataFrame): DataFrame.
        y_col (str): Name of the dependent column (target variable).
        x_cols (list): List of names of independent columns (predictor variables).

    Returns:
        statsmodels.regression.linear_model.RegressionResultsWrapper: Regression model results.
    """
    X = df[x_cols]
    y = df[y_col]
    X = sm.add_constant(X) # Add intercept
    model = sm.OLS(y, X).fit()
    return model

def perform_kmeans_clustering(df, feature_cols, n_clusters=3):
    """
    Performs K-Means clustering.

    Args:
        df (pandas.DataFrame): DataFrame.
        feature_cols (list): List of columns to use as features for clustering.
        n_clusters (int, optional): Number of clusters. Defaults to 3.

    Returns:
        pandas.DataFrame: DataFrame with a new 'cluster' column indicating cluster assignment for each row.
    """
    data_clustering = df[feature_cols].copy().dropna() # Copy to avoid modifying original, remove NaN for clustering
    if data_clustering.empty:
        print("Warning: Cannot perform clustering because there is no valid data in selected columns.")
        return df  # Return the original DataFrame unchanged

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_clustering)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # n_init to avoid warning
    df['cluster'] = pd.NA # Initialize with NA to maintain original length
    df.loc[data_clustering.index, 'cluster'] = kmeans.fit_predict(data_scaled).astype(str) # Assign clusters only to valid rows
    return df


if __name__ == '__main__':
    from data_loading import load_and_merge_data
    from data_cleaning import clean_data
    df_costs, _ = load_and_merge_data()
    df_costs_cleaned = clean_data(df_costs.copy())

    print("Statistical Analysis Example:")

    # Example of T-test
    t_stat, p_val_ttest = perform_ttest(df_costs_cleaned, 'state_name', 'mcsa', 'California', 'Texas')
    if t_stat is not None:
        print(f"\nT-test (MCSA between California and Texas):")
        print(f"  T-statistic: {t_stat:.4f}, P-value: {p_val_ttest:.4f}")
    else:
        print("\nCould not perform T-test (possibly missing data or empty groups).")


    # Example of ANOVA
    f_stat, p_val_anova = perform_anova(df_costs_cleaned, 'state_name', 'mcsa')
    if f_stat is not None:
        print(f"\nANOVA (MCSA by state):")
        print(f"  F-statistic: {f_stat:.4f}, P-value: {p_val_anova:.4f}")
    else:
        print("\nCould not perform ANOVA (possibly less than 2 valid groups).")


    # Example of Linear Regression
    regression_model = perform_linear_regression(df_costs_cleaned, 'mcsa', ['mhi_2018', 'unr_16'])
    print(f"\nLinear Regression (MCSA vs MHI and UNR):\n{regression_model.summary()}")


    # Example of K-Means Clustering
    df_clustered = perform_kmeans_clustering(df_costs_cleaned.copy(), ['mcsa', 'mhi_2018', 'unr_16'], n_clusters=4)
    print("\nDataFrame with K-Means clusters. First rows:")
    print(df_clustered[['county_name', 'state_name', 'cluster']].head())