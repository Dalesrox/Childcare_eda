
# Real run in: https://childcare-eda.streamlit.app/
# password: super_strong_password

# EDA on childcare costs data

Objetive: Implement Exploratory Data Analysis tecniques (EDA) in order to getting insights from childcare costs database (alvailable in Kaggle: https://www.kaggle.com/datasets/sujaykapadnis/childcare-costs).

## Introduction
The cost of childcare is a critical concern for many families around the world. With the rising cost of living and increasing demands on family budgets, understanding the factors that influence childcare costs is essential for policymakers, parents, and caregivers alike. The Childcare Costs dataset, available on Kaggle, provides a unique opportunity to explore this complex issue in depth.

This project aims to perform an exhaustive exploratory data analysis (EDA) on the Childcare Costs dataset, leveraging the power of Python, Pandas, and Seaborn to uncover hidden patterns, trends, and correlations. Through a combination of descriptive statistics, data cleaning techniques, and interactive visualizations, we will delve into the structure and characteristics of the data, identifying key variables, outliers, and relationships that can inform our understanding of childcare costs.

This EDA will encompass a range of techniques, including univariate and bivariate analysis, categorization of numeric data, and analysis of categories versus numeric data using sklearn exploratory techniques and Seaborn visualizations. By applying these methods, we will gain a deeper understanding of the factors that drive childcare costs, including the impact of demographic variables, geographic location, and service type.

The insights generated from this project will be invaluable for stakeholders seeking to optimize childcare resources, inform policy decisions, and support families in need. By showcasing the power of EDA in uncovering hidden patterns and trends, this project demonstrates the importance of data-driven decision-making in the social sector.

## Settings

- Run locally with:

```bash
streamlit run streamlit_app.py --server.address 127.0.0.1
```

- You can change the project password in src/.streamlit/secrets.toml, by default it is: super_strong_password