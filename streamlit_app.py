# run with streamlit run streamlit_app.py --server.address 127.0.0.1

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from src.data_loading import load_and_merge_data
from src.data_cleaning import clean_data
from src.eda_functions import histogram, scatter_plot, violin_plot, correlation_heatmap, create_categorical_columns
from src.statistical_analysis import (perform_ttest, perform_anova, perform_linear_regression,
                                    perform_kmeans_clustering, get_cluster_centroids, plot_linear_regression, perform_tukey_hsd,
                                    interpret_tukey_hsd_results)


# --- Data Loading and Initial Setup (Keep this part unchanged) ---
# Load data (this is done only once at the beginning of the app)
df_costs_raw, costs_dict = load_and_merge_data()
df_costs_cleaned = clean_data(df_costs_raw.copy())
df_costs_cleaned = create_categorical_columns(df_costs_cleaned)

numeric_options = df_costs_cleaned.select_dtypes(include=[np.number]).columns.tolist()
categorical_options_for_hue = ['None'] + ['cat_' + col for col in numeric_options] + ['state_name']
category_options = ['cat_' + i for i in numeric_options]

def get_variable_description(variable_name, dictionary_df):
    """Gets the description of a variable from the data dictionary DataFrame."""
    description_row = dictionary_df[dictionary_df['variable'] == variable_name.replace("cat_","")]
    if not description_row.empty:
        return description_row['description'].iloc[0]
    else:
        return "Select a variable!"

st.set_page_config(
    page_title="Interactive Exploration of Childcare Costs",
    page_icon="💰",  
    layout="centered",  
    initial_sidebar_state="expanded", 
    menu_items={
        'Report a bug': "https://github.com/Dalesrox/Childcare_eda/issues",
    }
)

# --- Authentication Function ---
def check_password():
    """Returns `True` if the user had the correct password."""    
    
    st.session_state["password"] = None
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]: # Uses st.secrets for secure password management
            st.session_state["password_correct"] = True
            st.session_state["password"] = None  # don't store password after it's correct
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for password.
        st.text_input(
            "**Easter egg -> Password is: super_strong_password**", placeholder="Password is super_strong_password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, display input and error.
        st.text_input(
            label="**Easter egg -> Password is: super_strong_password**", placeholder="Password is super_strong_password", type="password", on_change=password_entered, key="password"
        )
        st.error("password is super_strong_password")
        return False
    else:
        # Password correct.
        return True


st.title("EDA on Childcare Costs Data 💰")
if not check_password():
    st.subheader("Objetive:")
    st.text("""
        Implement Exploratory Data Analysis techniques (EDA) in order to get insights from the childcare costs database:\n 
    - https://www.kaggle.com/datasets/sujaykapadnis/childcare-costs.
\nAlso implement auth filter for login and logout in the website, for showing a way of secure the presentation.
\nThis project aims to perform an exhaustive exploratory data analysis (EDA) on the Childcare Costs dataset, leveraging the power of Python, Pandas, and Seaborn to uncover hidden patterns, trends, and correlations. Through a combination of descriptive statistics, data cleaning techniques, and interactive visualizations, we will delve into the structure and characteristics of the data, identifying key variables, outliers, and relationships that can inform our understanding of childcare costs.
\nThis EDA will encompass a range of techniques, including univariate and bivariate analysis, categorization of numeric data, and analysis of categories versus numeric data using sklearn exploratory techniques and Seaborn visualizations. By applying these methods, we will gain a deeper understanding of the factors that drive childcare costs, including the impact of demographic variables, geographic location, and service type.
""")
    st.stop()  

# --- If password is correct, continue with the app ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select the type of analysis:", ["Exploratory Data Analysis (EDA)", "Statistical Analysis"])

def logout_button():
    if st.sidebar.button("Logout"):
        st.session_state["password_correct"] = False
        st.rerun()

logout_button() 

if page == "Exploratory Data Analysis (EDA)":

    st.header("Exploratory Data Analysis (EDA) Visualizations")
    # --- Histogram ---
    st.subheader("Histogram")
    variable_hist = st.selectbox("Variable for Histogram", numeric_options, key='hist_var', index=numeric_options.index('mcsa'))
    st.write(f"**Variable Description:** {get_variable_description(variable_hist, costs_dict)}")
    if variable_hist:
        hist_fig = histogram(df_costs_cleaned, variable_hist)
        st.pyplot(hist_fig)

    # --- Scatter Plot ---
    st.subheader("Scatter Plot")
    scatter_x_variable = st.selectbox("X Variable for Scatter Plot", numeric_options, index=numeric_options.index('me_2018'), key='scatter_x')
    st.write(f"**Variable (X) Description:** {get_variable_description(scatter_x_variable, costs_dict)}")
    scatter_y_variable = st.selectbox("Y Variable for Scatter Plot", numeric_options, index=numeric_options.index('mcsa'), key='scatter_y')
    st.write(f"**Variable (Y) Description:** {get_variable_description(scatter_y_variable, costs_dict)}")
    scatter_hue_variable = st.selectbox("Variable for 'Hue' (color) in Scatter Plot (Optional)", categorical_options_for_hue, index=numeric_options.index('emp_service')+1, key='scatter_hue')
    st.write(f"**Variable (Hue) Description:** {get_variable_description(scatter_hue_variable, costs_dict)}, expressed in percentiles (0-25, 25-50, 50-75, 75-100)")
    if scatter_x_variable and scatter_y_variable:
        scatter_fig = scatter_plot(df_costs_cleaned, scatter_x_variable, scatter_y_variable, scatter_hue_variable if scatter_hue_variable != 'None' else None)
        st.pyplot(scatter_fig)

    # --- Violin Plot ---
    st.subheader("Violin Plot")
    violin_y_variable = st.selectbox("Y Variable for Violin Plot", numeric_options, index=numeric_options.index('mcsa'), key='violin_y')
    st.write(f"**Variable (Y) Description:** {get_variable_description(violin_y_variable, costs_dict)}")
    violin_x_variable = st.selectbox("X (Categorical) Variable for Violin Plot", category_options + ['state_name'], index=category_options.index('cat_me_2018'), key='violin_x') # Use categorical or state_name
    st.write(f"**Variable (X) Description:** {get_variable_description(violin_x_variable, costs_dict)}")
    violin_hue_variable = st.selectbox("Variable for 'Hue' (color) in Violin Plot (Optional)", categorical_options_for_hue, key='violin_hue')
    st.write(f"**Variable (Hue) Description:** {get_variable_description(violin_hue_variable, costs_dict)}, expressed in percentiles (0-25, 25-50, 50-75, 75-100)")

    if violin_y_variable and violin_x_variable:
        violin_fig = violin_plot(df_costs_cleaned, violin_y_variable, violin_x_variable, violin_hue_variable if violin_hue_variable != 'None' else None)
        st.pyplot(violin_fig)

    # --- Correlation Heatmap ---
    st.subheader("Correlation Heatmap")
    correlation_variable = st.selectbox("Select Variable for Correlation Heatmap", ['All Variables'] + numeric_options, index=numeric_options.index('mcsa')+1, key='correlation_var_selector') # Dropdown to select variable or all
    st.write(f"**Variable Description:** {get_variable_description(correlation_variable, costs_dict) if correlation_variable != 'All Variables' else 'Displays the correlation matrix for all numeric variables.'}") # Variable description, or general for 'All Variables'

    if st.checkbox("Show Correlation Heatmap", value=True): # Checkbox to enable/disable heatmap (can be computationally expensive)
        if correlation_variable == 'All Variables':
            heatmap_fig = correlation_heatmap(df_costs_raw) # Original heatmap for all variables
            st.pyplot(heatmap_fig)
        elif correlation_variable != 'All Variables' and correlation_variable: # If a specific variable is selected
            heatmap_fig = correlation_heatmap(df_costs_raw, target_variable=correlation_variable) # Calls modified heatmap function with target variable
            st.pyplot(heatmap_fig)


elif page == "Statistical Analysis":
    st.header("Statistical Analysis")

    # --- T-test ---
    st.subheader("Student's T-Test")

    ttest_variable_val = st.selectbox("Select Numeric Variable for T-Test", numeric_options, index=numeric_options.index('mcsa'))
    st.write(f"**Numeric Variable Description:** {get_variable_description(ttest_variable_val, costs_dict)}")

    grouping_category = st.selectbox("Select Categorical Variable for Grouping", category_options, index=category_options.index('cat_hispanic'))
    st.write(f"**Categorical Grouping Variable:** {get_variable_description(grouping_category, costs_dict)} expressed in percentiles (0-25, 25-50, 50-75, 75-100)")

    group_options = list(df_costs_cleaned[grouping_category].unique())
    if group_options:
        group1_ttest_val = st.selectbox(f"Select Group 1 ({grouping_category})", group_options, index=0)
        group2_ttest_val = st.selectbox(f"Select Group 2 ({grouping_category})", group_options, index=min(1, len(group_options)-1))

        if st.button("Run the T-Test!"):
            if group1_ttest_val == group2_ttest_val:
                st.error("Please select two different groups for the T-Test.")
            else:
                t_stat, p_val_ttest = perform_ttest(df_costs_cleaned, grouping_category, ttest_variable_val, group1_ttest_val, group2_ttest_val)
                if t_stat is not None:
                    st.write(f"**T-Test Results ({ttest_variable_val} for {grouping_category} = '{group1_ttest_val}' vs '{group2_ttest_val}'):**")
                    st.write(f"T-Statistic: {t_stat:.4f}, P-Value: {p_val_ttest:.4f}")
                    if p_val_ttest < 0.05:
                        st.success("Significant difference found (p < 0.05).")
                    else:
                        st.info("No significant difference found (p >= 0.05).")
                else:
                    st.error("T-Test could not be performed (possibly missing data or the groups have no data).")
    else:
        st.warning(f"No groups available for the T-Test based on the selected Categorical Variable: '{grouping_category}'.")


    st.subheader("ANOVA (Analysis of Variance)")

    anova_variable_val = st.selectbox("Select Numeric Variable for ANOVA", numeric_options, index=numeric_options.index('mcsa'))
    st.write(f"**Numeric Variable Description:** {get_variable_description(anova_variable_val, costs_dict)}")

    anova_grouping_category = st.selectbox("Select Categorical Variable for Grouping (ANOVA)", category_options, index=category_options.index('cat_hispanic'))
    st.write(f"**Categorical Grouping Variable:** {get_variable_description(anova_grouping_category, costs_dict)} expressed in percentiles (0-25, 25-50, 50-75, 75-100)")


    if st.button("Perform ANOVA!"):
        anova_table_result = perform_anova(df_costs_cleaned, anova_grouping_category, anova_variable_val)
        if anova_table_result is not None:
            st.write(f"**ANOVA Results ({anova_variable_val} by {anova_grouping_category}):**")
            st.dataframe(round(anova_table_result))
            f_statistic = anova_table_result['F'].iloc[0]
            p_value_anova = anova_table_result['PR(>F)'].iloc[0]

            st.write(f"F-Statistic: {f_statistic:.4f}, P-Value: {p_value_anova:.4f}")
            if p_value_anova < 0.05:
                st.success("Significant differences found between groups (p < 0.05).")
                # Perform and display Tukey's HSD post-hoc test
                tukey_result_df = perform_tukey_hsd(df_costs_cleaned, anova_grouping_category, anova_variable_val)
                if tukey_result_df is not None:
                    st.subheader("Tukey's HSD Post-Hoc Test")
                    st.dataframe(round(tukey_result_df,4)) # Display Tukey's HSD results
                    st.write("Pairwise comparisons using Tukey's HSD post-hoc test. 'reject' = True indicates a significant difference between the means of the two groups at a significance level of 0.05.")
                    # Interpret and display Tukey's HSD results as text
                    tukey_interpretation_text = interpret_tukey_hsd_results(tukey_result_df, anova_grouping_category, anova_variable_val)
                    st.write(tukey_interpretation_text)
            else:
                st.info("No evidence of significant differences between groups (p >= 0.05).")
        else:
            st.error("ANOVA could not be performed (possibly fewer than 2 valid groups or calculation error). Check the console for details.")


    # --- Linear Regression ---
    st.subheader("Linear Regression")
    y_var_reg_val = st.selectbox("Dependent Variable (Y) for Regression", numeric_options, index=numeric_options.index('mcsa'))
    x_vars_reg_val = st.multiselect("Independent Variables (X) for Regression", numeric_options, default=['mhi_2018'])
    for item in x_vars_reg_val:
        st.write(f"**Variable {item}:** {get_variable_description(item, costs_dict)}")

    if st.button("Perform Linear Regression!"):
        if y_var_reg_val and x_vars_reg_val:
            regression_model = perform_linear_regression(df_costs_cleaned, y_var_reg_val, x_vars_reg_val)
            st.write("**Linear Regression Model Summary:**")
            st.write(regression_model.summary())

            if len(x_vars_reg_val) == 1: # Plot only for simple linear regression
                try:
                    reg_fig = plot_linear_regression(regression_model, df_costs_cleaned, y_var_reg_val, x_vars_reg_val)
                    st.subheader("Linear Regression Plot")
                    st.pyplot(reg_fig)
                except Exception as e:
                    st.error(f"Could not plot the linear regression: {e}")
            elif len(x_vars_reg_val) > 1:
                st.info("Linear regression plot is only available for simple linear regression (one independent variable).")

        else:
            st.warning("Please select a Dependent Variable (Y) and at least one Independent Variable (X) for regression.")


    # --- K-Means Clustering ---
    st.subheader("K-Means Clustering")
    cluster_features_val = st.multiselect("Features for K-Means Clustering", numeric_options, default=['mcsa', 'hispanic', 'emp_m'])
    n_clusters_input_val = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=3)
    for item in cluster_features_val:
        st.write(f"**Variable {item}:** {get_variable_description(item, costs_dict)}")

    if st.button("Start clustering!"):
        if cluster_features_val:
            df_clustered = perform_kmeans_clustering(df_costs_cleaned.copy(), cluster_features_val, n_clusters=n_clusters_input_val)

            centroids_df = get_cluster_centroids(df_clustered, cluster_features_val)
            st.subheader("Cluster Centroids")
            st.dataframe(centroids_df.round())

            data_clustering = df_costs_cleaned[cluster_features_val].copy().dropna()
            if not data_clustering.empty:

                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_clustering)
                labels = pd.DataFrame(df_clustered.loc[data_clustering.index, 'cluster'].astype(int))

                # Silhouette Score (Sample)
                sample_size = min(len(data_scaled), 1000)
                sampled_indices = np.random.choice(len(data_scaled), sample_size, replace=False)
                silhouette_avg = silhouette_score(data_scaled[sampled_indices], labels.loc[labels.index[sampled_indices]])

                # Cohesion and Separability
                scaled_centroids = scaler.transform(centroids_df[cluster_features_val])
                centroids_df_scaled = pd.DataFrame(scaled_centroids, columns=cluster_features_val)
                centroids_df_scaled['cluster'] = centroids_df['cluster']

                # Metrics Table
                metrics_df = pd.DataFrame({
                    'Metric': ['Silhouette (Sample)'],
                    'Value': [silhouette_avg]
                })
                st.subheader("Clustering Metrics")
                st.dataframe(metrics_df.round(3))


        # Visualize Clusters (simple scatter plot if 2 features, more complex for >2D)
        if len(cluster_features_val) >= 2:
            st.subheader("Cluster Visualization (First 2 Features)") # Limitation: only the first 2 vars for 2D plot
            cluster_scatter_fig = plt.figure()
            sns.scatterplot(data=df_clustered, x=cluster_features_val[0], y=cluster_features_val[1], hue='cluster', palette='viridis')
            plt.title('K-Means Clusters')
            st.pyplot(cluster_scatter_fig)

        else:
            st.warning("Please select at least one feature for K-Means Clustering.")


st.write("---")
st.write("Developed by Leonardo Espinosa")