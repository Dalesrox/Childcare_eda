import pandas as pd

def clean_data(df_costs):
    """
    Cleans the childcare costs DataFrame:
    - Removes duplicates.
    - Removes rows with NaN values.

    Args:
        df_costs (pandas.DataFrame): Childcare costs DataFrame.

    Returns:
        pandas.DataFrame: Cleaned DataFrame.
    """
    df_costs = df_costs.drop_duplicates()
    df_costs = df_costs.dropna() # Remove NaN values

    # Optional: Convert certain columns to category type if needed for memory/speed
    # df_costs['county_name'] = df_costs['county_name'].astype('category')
    # df_costs['state_name'] = df_costs['state_name'].astype('category')

    return df_costs

if __name__ == '__main__':
    from data_loading import load_and_merge_data
    df_costs, _ = load_and_merge_data() # Load data
    initial_shape = df_costs.shape[0]
    df_costs_cleaned = clean_data(df_costs.copy()) # Clean a copy to avoid modifying the original
    cleaned_shape = df_costs_cleaned.shape[0]
    print("Data cleaning completed.")
    print(f"Rows before cleaning: {initial_shape}")
    print(f"Rows after cleaning: {cleaned_shape}")