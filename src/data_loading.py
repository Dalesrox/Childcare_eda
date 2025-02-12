import pandas as pd

def load_and_merge_data(costs_file='./data/childcare_costs.csv', counties_file='./data/counties.csv', dict_file='./data/childcare_dictionary.csv'):
    """
    Loads childcare costs, counties, and dictionary datasets and merges them.
    Args:
        costs_file (str, optional): Filename for childcare costs CSV. Defaults to 'childcare_costs.csv'.
        counties_file (str, optional): Filename for counties data CSV. Defaults to 'counties.csv'.
        dict_file (str, optional): Filename for data dictionary CSV. Defaults to 'childcare_dictionary.csv'.
    Returns:
        pandas.DataFrame: Merged DataFrame with childcare costs and counties data.
        pandas.DataFrame: Data dictionary DataFrame.
    """
    df_costs = pd.read_csv(costs_file)
    df_county = pd.read_csv(counties_file)
    costs_dict = pd.read_csv(dict_file, sep=';', on_bad_lines='skip')
    # Merge county data into the childcare costs dataframe
    df_costs = pd.merge(df_costs, df_county[['county_fips_code', 'county_name', 'state_name']], on='county_fips_code')
    df_costs['county_name'] = df_costs['county_name'].astype('category')
    df_costs['state_name'] = df_costs['state_name'].astype('category')
    return df_costs, costs_dict

if __name__ == '__main__':
    df_costs, costs_dict = load_and_merge_data()
    print("Data loaded and merged.")
    print(f"Shape of df_costs: {df_costs.shape}")
    print(f"Shape of costs_dict: {costs_dict.shape}")