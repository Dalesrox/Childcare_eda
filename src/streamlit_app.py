import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_loading import load_and_merge_data
from data_cleaning import clean_data
from eda_functions import boxplot, histogram, scatter_plot, violin_plot, correlation_heatmap, create_categorical_columns
from statistical_analysis import perform_ttest, perform_anova, perform_linear_regression, perform_kmeans_clustering


# Load data
df_costs_raw, costs_dict = load_and_merge_data()
df_costs_cleaned = clean_data(df_costs_raw.copy()) # Use copy for cleaning
df_costs_cleaned = create_categorical_columns(df_costs_cleaned) # Create categorical columns

numeric_options = df_costs_cleaned.select_dtypes(include=[np.number]).columns.tolist()
categorical_options_for_hue = ['None'] + ['cat_' + col for col in numeric_options] + ['state_name'] # Options for 'hue'
category_options = ['cat_' + i for i in numeric_options] # Created categorical options

def get_variable_description(variable_name, dictionary_df):
    """Retrieves the description of a variable from the data dictionary DataFrame."""
    description_row = dictionary_df[dictionary_df['variable'] == variable_name.replace("cat_","")]
    if not description_row.empty:
        return description_row['description'].iloc[0]
    else:
        return "Select a variable!"

st.title("Interactive Childcare Costs Exploration")

# --- EDA Visualizations Section ---
st.header("Exploratory Data Analysis (EDA) Visualizations")

# --- Boxplot ---
st.subheader("Boxplot")
boxplot_variable = st.selectbox("Variable for Boxplot", numeric_options, key='boxplot_var', index=numeric_options.index('mcsa'))
st.write(f"**Variable Description:** {get_variable_description(boxplot_variable, costs_dict)}")
if boxplot_variable:
    boxplot_fig = boxplot(df_costs_cleaned, boxplot_variable)
    st.pyplot(boxplot_fig)


# --- Histogram ---
st.subheader("Histogram")
variable_hist = st.selectbox("Variable for Histogram", numeric_options, key='hist_var', index=numeric_options.index('mcsa'))
st.write(f"**Variable Description:** {get_variable_description(variable_hist, costs_dict)}")
if variable_hist:
    # No need to create plt.figure() here, histogram function will handle it
    hist_fig = histogram(df_costs_cleaned, variable_hist)
    st.pyplot(hist_fig)

# --- Scatter Plot ---
st.subheader("Scatter Plot")
scatter_x_variable = st.selectbox("Variable X for Scatter Plot", numeric_options, index=numeric_options.index('pr_f'), key='scatter_x')
st.write(f"**Variable (X) Description:** {get_variable_description(scatter_x_variable, costs_dict)}")
scatter_y_variable = st.selectbox("Variable Y for Scatter Plot", numeric_options, index=numeric_options.index('mcsa'), key='scatter_y')
st.write(f"**Variable (Y) Description:** {get_variable_description(scatter_y_variable, costs_dict)}")
scatter_hue_variable = st.selectbox("Variable for 'Hue' (color) in Scatter Plot (Optional)", categorical_options_for_hue, index=numeric_options.index('unr_20to64')+1, key='scatter_hue')
st.write(f"**Variable (Hue) Description:** {get_variable_description(scatter_hue_variable, costs_dict)}, expresed in percentiles (0-25, 25-50, 50-75, 75-100)")
if scatter_x_variable and scatter_y_variable:
    # No need to create plt.figure() here, scatter_plot function will handle it
    scatter_fig = scatter_plot(df_costs_cleaned, scatter_x_variable, scatter_y_variable, scatter_hue_variable if scatter_hue_variable != 'None' else None)
    st.pyplot(scatter_fig)


# --- Violin Plot ---
st.subheader("Violin Plot")
violin_y_variable = st.selectbox("Variable Y for Violin Plot", numeric_options, index=numeric_options.index('mcsa'), key='violin_y')
st.write(f"**Variable (Y) Description:** {get_variable_description(violin_y_variable, costs_dict)}")
violin_x_variable = st.selectbox("Variable X (Categorical) for Violin Plot", category_options + ['state_name'], index=category_options.index('cat_hispanic') if 'cat_hispanic' in category_options else 0, key='violin_x') # Use categorical or state_name
st.write(f"**Variable (X) Description:** {get_variable_description(violin_x_variable, costs_dict)}")
violin_hue_variable = st.selectbox("Variable for 'Hue' (color) in Violin Plot (Optional)", categorical_options_for_hue, key='violin_hue')
st.write(f"**Variable (Hue) Description:** {get_variable_description(scatter_hue_variable, costs_dict)}, expresed in percentiles (0-25, 25-50, 50-75, 75-100)")

if violin_y_variable and violin_x_variable:
    # No need to create plt.figure() here, violin_plot function will handle it
    violin_fig = violin_plot(df_costs_cleaned, violin_y_variable, violin_x_variable, violin_hue_variable if violin_hue_variable != 'None' else None)
    st.pyplot(violin_fig)


# --- Correlation Heatmap ---
st.subheader("Correlation Heatmap")
correlation_variable = st.selectbox("Select Variable for Correlation Heatmap", ['All Variables'] + numeric_options, index=numeric_options.index('mcsa')+1, key='correlation_var_selector') # Dropdown to select single variable or all
st.write(f"**Variable Description:** {get_variable_description(correlation_variable, costs_dict) if correlation_variable != 'All Variables' else 'Displays correlation matrix for all numeric variables.'}") # Display description, or general description for 'All Variables'

if st.checkbox("Show Correlation Heatmap", value=True): # Checkbox to activate/deactivate heatmap (can be computationally expensive)
    if correlation_variable == 'All Variables':
        heatmap_fig = correlation_heatmap(df_costs_raw) # Original heatmap for all variables
        st.pyplot(heatmap_fig)
    elif correlation_variable != 'All Variables' and correlation_variable: # If a specific variable is selected
        heatmap_fig = correlation_heatmap(df_costs_raw, target_variable=correlation_variable) # Call modified heatmap function with target variable
        st.pyplot(heatmap_fig)


# --- Statistical Analysis Section ---
st.header("Statistical Analysis")

# --- T-test ---
st.subheader("Student's T-test")

# Selector 1: Variable para T-test
ttest_variable_val = st.selectbox("Select Numeric Variable for T-test", numeric_options, index=numeric_options.index('mcsa'))
st.write(f"**Numeric Variable Description:** {get_variable_description(ttest_variable_val, costs_dict)}")

# Selector 2: Categorical Variable for Grouping
grouping_category = st.selectbox("Select Categorical Variable for Grouping", category_options, index=category_options.index('cat_flfpr_20to64_under6'))
st.write(f"**Grouping Categorical Variable:** {get_variable_description(grouping_category, costs_dict)} expresed in percentiles (0-25, 25-50, 50-75, 75-100)")

# Get dynamic group options based on the selected categorical column
group_options = list(df_costs_cleaned[grouping_category].unique())
if group_options: # Ensure groups are available
    # Selectors 3 & 4: Groups to compare, dynamic based on selected categorical column
    group1_ttest_val = st.selectbox(f"Select Group 1 ({grouping_category})", group_options, index=0) # Default to first group
    group2_ttest_val = st.selectbox(f"Select Group 2 ({grouping_category})", group_options, index=min(1, len(group_options)-1)) # Default to second group, or first if only one group


    if st.button("Perform T-test"):
        if group1_ttest_val == group2_ttest_val:
            st.error("Please select two different groups for T-test.")
        else:
            t_stat, p_val_ttest = perform_ttest(df_costs_cleaned, grouping_category, ttest_variable_val, group1_ttest_val, group2_ttest_val)
            if t_stat is not None:
                st.write(f"**T-test Results ({ttest_variable_val} for {grouping_category} = '{group1_ttest_val}' vs '{group2_ttest_val}'):**")
                st.write(f"T-statistic: {t_stat:.4f}, P-value: {p_val_ttest:.4f}")
                if p_val_ttest < 0.05:
                    st.success("Significant difference found (p < 0.05).")
                else:
                    st.info("No significant difference found (p >= 0.05).")
            else:
                st.error("Could not perform T-test (possibly missing data or groups have no data).")
else:
    st.warning(f"No groups available for T-test based on selected Categorical Variable: '{grouping_category}'.")


st.subheader("ANOVA (Analysis of Variance)")

# Selector 1: Variable para ANOVA (Numeric)
anova_variable_val = st.selectbox("Select Numeric Variable for ANOVA", numeric_options, index=numeric_options.index('mcsa'))
st.write(f"**Numeric Variable Description:** {get_variable_description(anova_variable_val, costs_dict)}")

# Selector 2: Categorical Variable for Grouping (same as T-test for consistency, but ANOVA can handle more groups)
anova_grouping_category = st.selectbox("Select Categorical Variable for Grouping (ANOVA)", category_options, index=category_options.index('cat_flfpr_20to64_under6'))
st.write(f"**Grouping Categorical Variable:** {get_variable_description(anova_grouping_category, costs_dict)} expresed in percentiles (0-25, 25-50, 50-75, 75-100)")



if st.button("Perform ANOVA"):
    anova_table_result = perform_anova(df_costs_cleaned, anova_grouping_category, anova_variable_val) # Obtiene la tabla ANOVA completa
    if anova_table_result is not None:
        st.write(f"**ANOVA Results ({anova_variable_val} by {anova_grouping_category}):**")
        st.dataframe(anova_table_result) # Muestra la tabla ANOVA usando st.dataframe
        # Adicionalmente, podrías querer mostrar F-statistic y p-value de la tabla (opcional):
        f_statistic = anova_table_result['F'].iloc[0]
        p_value_anova = anova_table_result['PR(>F)'].iloc[0]

        st.write(f"F-statistic: {f_statistic:.4f}, P-value: {p_value_anova:.4f}") # Muestra F y P-value debajo de la tabla
        if p_value_anova < 0.05:
            st.success("Significant differences among groups found (p < 0.05).")
        else:
            st.info("No evidence of significant differences among groups (p >= 0.05).")
    else:
        st.error("Could not perform ANOVA (possibly less than 2 valid groups or error in calculation). See console for details.")


# --- Linear Regression ---
st.subheader("Linear Regression")
y_var_reg_val = st.selectbox("Dependent Variable (Y) for Regression", numeric_options, index=numeric_options.index('mcsa'))
x_vars_reg_val = st.multiselect("Independent Variables (X) for Regression", numeric_options, default=['mhi_2018', 'unr_16']) # Multiselect for multiple X

if st.button("Perform Linear Regression"):
    if y_var_reg_val and x_vars_reg_val:
        regression_model = perform_linear_regression(df_costs_cleaned, y_var_reg_val, x_vars_reg_val)
        st.write("**Linear Regression Model Summary:**")
        st.write(regression_model.summary())
    else:
        st.warning("Please select a Dependent Variable (Y) and at least one Independent Variable (X) for regression.")


# --- K-Means Clustering ---
st.subheader("K-Means Clustering")
cluster_features_val = st.multiselect("Features for K-Means Clustering", numeric_options, default=['mcsa', 'mhi_2018', 'unr_16'])
n_clusters_input_val = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3) # Slider for number of clusters

if st.button("Perform K-Means Clustering"):
    if cluster_features_val:
        df_clustered = perform_kmeans_clustering(df_costs_cleaned.copy(), cluster_features_val, n_clusters=n_clusters_input_val) # Copy to avoid modifying df_costs_cleaned
        st.write("**DataFrame with Clusters (First Rows):**")
        st.dataframe(df_clustered[['county_name', 'state_name', 'cluster']].head())

        # Visualize Clusters (simple scatter plot if 2 features, more complex for >2D visualization)
        if len(cluster_features_val) >= 2:
            st.subheader("Cluster Visualization (First 2 Features)") # Limitation: only first 2 vars for 2D scatterplot
            cluster_scatter_fig = plt.figure()
            sns.scatterplot(data=df_clustered, x=cluster_features_val[0], y=cluster_features_val[1], hue='cluster', palette='viridis')
            plt.title('K-Means Clusters')
            st.pyplot(cluster_scatter_fig)

    else:
        st.warning("Please select at least one feature for K-Means Clustering.")


st.write("---")
st.write("Developed with Streamlit")