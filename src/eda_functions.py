import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def histogram(df, variable):
    """Generates a histogram for a specific variable in the DataFrame."""
    fig = plt.figure(figsize=(8, 6)) # Create figure object and return it
    sns.histplot(df[variable], kde=True)
    plt.title(f'Histogram of {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frequency')
    return fig # Return the figure

def scatter_plot(df, variable1, variable2, variable3=None):
    """Generates a scatter plot with an option for a third categorical variable for color."""
    joint_plot = sns.jointplot(data=df, x=variable1, y=variable2, kind="scatter", hue=variable3 if variable3 else None) # Create jointplot object
    joint_plot.figure.suptitle(f'Scatter Plot: {variable1} vs {variable2}', y=1.02) # Set suptitle using the jointplot.fig
    return joint_plot.figure # Return the figure from the jointplot

def violin_plot(df, variable1, variable2, variable3=None):
    """Generates a violin plot with an option for a third categorical variable for 'hue'."""
    fig = plt.figure(figsize=(10,6)) # Create figure object and return it
    if variable3:
        sns.violinplot(data=df, y=variable1, x=variable2, hue=variable3, split=False, inner="quart")
    else:
        sns.violinplot(data=df, y=variable1, x=variable2, inner="quart")
    plt.title(f'Violin Plot: {variable1} by {variable2}')
    return fig # Return the figure

def correlation_heatmap(df, columns_to_drop=None, target_variable=None):
    """Generates a correlation heatmap for the DataFrame.
    If target_variable is specified, it shows correlation of target_variable
    with all other numeric variables. Otherwise, shows full correlation matrix.
    """
    if columns_to_drop is None:
        columns_to_drop = ['county_name', 'state_name'] # Default categorical columns

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in columns_to_drop] # exclude columns_to_drop

    if target_variable and target_variable in numeric_cols: # if target_variable is specified and is numeric column
        other_numeric_cols = [col for col in numeric_cols if col != target_variable] # All other numeric columns
        if other_numeric_cols: # proceed if there are other numeric columns to correlate with
            correlation_series = df[other_numeric_cols].corrwith(df[target_variable]).to_frame(name='correlation') # Calculate correlation of target variable with others
            df_corr = correlation_series.sort_values(by='correlation', ascending=False) # Sort by correlation value for better visualization
            fig_height = max(2, len(other_numeric_cols) * 0.3) # Adjust figure height dynamically based on number of variables, minimum height of 2
            fig = plt.figure(figsize=(8, fig_height)) # Adjust figure height dynamically and keep width fixed at 8
            sns.heatmap(df_corr, annot=True, cmap='RdYlGn', linewidths=0.2, fmt=".2f") # heatmap for series (single column)
            plt.title(f'Correlation with {target_variable}')
            plt.yticks(rotation=0) # Keep y-tick labels horizontal for readability
            return fig
        else: # No other numeric columns to correlate with
            fig = plt.figure(figsize=(6,2)) # small empty plot
            plt.text(0.5, 0.5, 'No other numeric columns to correlate with the selected variable.', horizontalalignment='center', verticalalignment='center') # Informative message in heatmap area
            plt.axis('off') # No axes needed
            return fig

    else: # Original heatmap for all variables if target_variable is None or 'All Variables' selected
        df_corr = df[numeric_cols].corr()
        fig = plt.figure(figsize=(20,15)) # Reduced size for full heatmap, original was (30,25) and too large
        sns.heatmap(round(df_corr, 2), annot=True, cmap='RdYlGn', linewidths=0.2)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.title('Correlation Heatmap (All Variables)')
        return fig



def create_categorical_columns(df):
    """Creates categorical columns based on quartiles of numeric columns."""
    numeric_options = df.select_dtypes(include=[np.number]).columns.tolist()
    descriptive = df.describe()

    def to_categorical(column, descriptive_stats):
        bins = [-np.inf, descriptive_stats.loc['25%', column], descriptive_stats.loc['50%', column], descriptive_stats.loc['75%', column], np.inf]
        if len(set(bins)) == len(bins): # Check for repeated bin values
            return pd.cut(df[column], bins=bins, labels=['p0-p25', 'p25-p50', 'p50-p75', 'p75-p100'], ordered=False)
        else: # Create bins by range if quartile bins are not unique
            return pd.cut(df[column], bins=3, labels=['low', 'mid', 'high'], ordered=False)

    for column in numeric_options:
        df['cat_' + column] = to_categorical(column=column, descriptive_stats=descriptive)

    return df


if __name__ == '__main__':
    from data_loading import load_and_merge_data
    from data_cleaning import clean_data
    df_costs, _ = load_and_merge_data()
    df_costs_cleaned = clean_data(df_costs.copy()) # Use the cleaned version for EDA

    print("EDA Functions Example:")

    # Example of boxplot (you can uncomment to run)
    # fig_boxplot = boxplot(df_costs_cleaned, 'mcsa') # Capture figure
    # plt.show() # Show in interactive window if running directly

    # Example of histogram
    fig_hist = histogram(df_costs_cleaned, 'mhi_2018') # Capture figure
    plt.show() # Show in interactive window if running directly

    # Example of scatter plot
    fig_scatter = scatter_plot(df_costs_cleaned, 'unr_16', 'mhi_2018') # Capture figure
    plt.show() # Show in interactive window if running directly


    # Example of violin plot
    fig_violin = violin_plot(df_costs_cleaned, 'mcsa', 'state_name') # Capture figure
    plt.show() # Show in interactive window if running directly

    # Example of correlation heatmap
    fig_heatmap = correlation_heatmap(df_costs_cleaned) # Capture figure
    plt.show() # Show in interactive window if running directly


    # Example of creating categorical columns
    df_costs_categorical = create_categorical_columns(df_costs_cleaned.copy())
    print("\nDataFrame with categorical columns created. First rows:")
    print(df_costs_categorical.head())