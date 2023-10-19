# EDA on childcare costs data

Objetive: Implement Exploratory Data Analysis tecniques (EDA) in order to getting insights from childcare costs database (alvailable in Kaggle: https://www.kaggle.com/datasets/sujaykapadnis/childcare-costs).



```python
#import libaries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ipywidgets as widgets

pd.set_option('display.max_columns', None)
```

## Data Reading

We will work with 3 datasets:

* childcare_costs.csv: dataset with information about unemployment, childcare and poverty per county.
* counties.csv: dataset with counties data such as name and state.
* childcare_dictionary: dataset built from Kaggle's website table (https://www.kaggle.com/datasets/sujaykapadnis/childcare-costs)


```python
#Read data
df_costs = pd.read_csv('childcare_costs.csv')
df_county = pd.read_csv('counties.csv')
costs_dict = pd.read_csv('childcare_dictionary.csv', sep=';', on_bad_lines='skip')

print("Shapes of data (rows, columns):\n\n1. childcare_costs:", df_costs.shape, "\n2. counties:", df_county.shape, "\n3. dict:", costs_dict.shape)
```

    Shapes of data (rows, columns):
    
    1. childcare_costs: (34567, 61) 
    2. counties: (3144, 4) 
    3. dict: (61, 3)
    

## Check datatypes


```python
#We have 61 columns: 51 float and 10 int64 (as expected from source metadata)
#Also we can see some columns with null data
print(df_costs.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 34567 entries, 0 to 34566
    Data columns (total 61 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   county_fips_code           34567 non-null  int64  
     1   study_year                 34567 non-null  int64  
     2   unr_16                     34567 non-null  float64
     3   funr_16                    34567 non-null  float64
     4   munr_16                    34567 non-null  float64
     5   unr_20to64                 34567 non-null  float64
     6   funr_20to64                34567 non-null  float64
     7   munr_20to64                34567 non-null  float64
     8   flfpr_20to64               34567 non-null  float64
     9   flfpr_20to64_under6        34567 non-null  float64
     10  flfpr_20to64_6to17         34567 non-null  float64
     11  flfpr_20to64_under6_6to17  34567 non-null  float64
     12  mlfpr_20to64               34567 non-null  float64
     13  pr_f                       34567 non-null  float64
     14  pr_p                       34567 non-null  float64
     15  mhi_2018                   34567 non-null  float64
     16  me_2018                    34567 non-null  float64
     17  fme_2018                   34567 non-null  float64
     18  mme_2018                   34567 non-null  float64
     19  total_pop                  34567 non-null  int64  
     20  one_race                   34567 non-null  float64
     21  one_race_w                 34567 non-null  float64
     22  one_race_b                 34567 non-null  float64
     23  one_race_i                 34567 non-null  float64
     24  one_race_a                 34567 non-null  float64
     25  one_race_h                 34567 non-null  float64
     26  one_race_other             34567 non-null  float64
     27  two_races                  34567 non-null  float64
     28  hispanic                   34567 non-null  float64
     29  households                 34567 non-null  int64  
     30  h_under6_both_work         34567 non-null  int64  
     31  h_under6_f_work            34567 non-null  int64  
     32  h_under6_m_work            34567 non-null  int64  
     33  h_under6_single_m          34565 non-null  float64
     34  h_6to17_both_work          34567 non-null  int64  
     35  h_6to17_fwork              34567 non-null  int64  
     36  h_6to17_mwork              34567 non-null  int64  
     37  h_6to17_single_m           34565 non-null  float64
     38  emp_m                      34567 non-null  float64
     39  memp_m                     34567 non-null  float64
     40  femp_m                     34567 non-null  float64
     41  emp_service                34567 non-null  float64
     42  memp_service               34567 non-null  float64
     43  femp_service               34567 non-null  float64
     44  emp_sales                  34567 non-null  float64
     45  memp_sales                 34567 non-null  float64
     46  femp_sales                 34567 non-null  float64
     47  emp_n                      34567 non-null  float64
     48  memp_n                     34567 non-null  float64
     49  femp_n                     34567 non-null  float64
     50  emp_p                      34567 non-null  float64
     51  memp_p                     34567 non-null  float64
     52  femp_p                     34567 non-null  float64
     53  mcsa                       23593 non-null  float64
     54  mfccsa                     23383 non-null  float64
     55  mc_infant                  23593 non-null  float64
     56  mc_toddler                 23593 non-null  float64
     57  mc_preschool               23593 non-null  float64
     58  mfcc_infant                23383 non-null  float64
     59  mfcc_toddler               23383 non-null  float64
     60  mfcc_preschool             23383 non-null  float64
    dtypes: float64(51), int64(10)
    memory usage: 16.1 MB
    None
    


```python
#Here we have an ID and county data with non-null data
print(df_county.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3144 entries, 0 to 3143
    Data columns (total 4 columns):
     #   Column              Non-Null Count  Dtype 
    ---  ------              --------------  ----- 
     0   county_fips_code    3144 non-null   int64 
     1   county_name         3144 non-null   object
     2   state_name          3144 non-null   object
     3   state_abbreviation  3144 non-null   object
    dtypes: int64(1), object(3)
    memory usage: 98.4+ KB
    None
    


```python
#Check data head (we can see all columns because of pd.set_option('display.max_columns', None) row at the begining)
df_costs.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>county_fips_code</th>
      <th>study_year</th>
      <th>unr_16</th>
      <th>funr_16</th>
      <th>munr_16</th>
      <th>unr_20to64</th>
      <th>funr_20to64</th>
      <th>munr_20to64</th>
      <th>flfpr_20to64</th>
      <th>flfpr_20to64_under6</th>
      <th>flfpr_20to64_6to17</th>
      <th>flfpr_20to64_under6_6to17</th>
      <th>mlfpr_20to64</th>
      <th>pr_f</th>
      <th>pr_p</th>
      <th>mhi_2018</th>
      <th>me_2018</th>
      <th>fme_2018</th>
      <th>mme_2018</th>
      <th>total_pop</th>
      <th>one_race</th>
      <th>one_race_w</th>
      <th>one_race_b</th>
      <th>one_race_i</th>
      <th>one_race_a</th>
      <th>one_race_h</th>
      <th>one_race_other</th>
      <th>two_races</th>
      <th>hispanic</th>
      <th>households</th>
      <th>h_under6_both_work</th>
      <th>h_under6_f_work</th>
      <th>h_under6_m_work</th>
      <th>h_under6_single_m</th>
      <th>h_6to17_both_work</th>
      <th>h_6to17_fwork</th>
      <th>h_6to17_mwork</th>
      <th>h_6to17_single_m</th>
      <th>emp_m</th>
      <th>memp_m</th>
      <th>femp_m</th>
      <th>emp_service</th>
      <th>memp_service</th>
      <th>femp_service</th>
      <th>emp_sales</th>
      <th>memp_sales</th>
      <th>femp_sales</th>
      <th>emp_n</th>
      <th>memp_n</th>
      <th>femp_n</th>
      <th>emp_p</th>
      <th>memp_p</th>
      <th>femp_p</th>
      <th>mcsa</th>
      <th>mfccsa</th>
      <th>mc_infant</th>
      <th>mc_toddler</th>
      <th>mc_preschool</th>
      <th>mfcc_infant</th>
      <th>mfcc_toddler</th>
      <th>mfcc_preschool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>2008</td>
      <td>5.42</td>
      <td>4.41</td>
      <td>6.32</td>
      <td>4.6</td>
      <td>3.5</td>
      <td>5.6</td>
      <td>68.9</td>
      <td>66.9</td>
      <td>79.59</td>
      <td>60.81</td>
      <td>84.0</td>
      <td>8.5</td>
      <td>11.5</td>
      <td>58462.55</td>
      <td>32710.60</td>
      <td>25156.25</td>
      <td>41436.80</td>
      <td>49744</td>
      <td>98.1</td>
      <td>78.9</td>
      <td>17.7</td>
      <td>0.4</td>
      <td>0.4</td>
      <td>0.0</td>
      <td>0.7</td>
      <td>1.9</td>
      <td>1.8</td>
      <td>18373</td>
      <td>1543</td>
      <td>970</td>
      <td>22</td>
      <td>995.0</td>
      <td>4900</td>
      <td>1308</td>
      <td>114</td>
      <td>1966.0</td>
      <td>27.40</td>
      <td>24.41</td>
      <td>30.68</td>
      <td>17.06</td>
      <td>15.53</td>
      <td>18.75</td>
      <td>29.11</td>
      <td>15.97</td>
      <td>43.52</td>
      <td>13.21</td>
      <td>22.54</td>
      <td>2.99</td>
      <td>13.22</td>
      <td>21.55</td>
      <td>4.07</td>
      <td>80.92</td>
      <td>81.40</td>
      <td>104.95</td>
      <td>104.95</td>
      <td>85.92</td>
      <td>83.45</td>
      <td>83.45</td>
      <td>81.40</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1001</td>
      <td>2009</td>
      <td>5.93</td>
      <td>5.72</td>
      <td>6.11</td>
      <td>4.8</td>
      <td>4.6</td>
      <td>5.0</td>
      <td>70.8</td>
      <td>63.7</td>
      <td>78.41</td>
      <td>59.91</td>
      <td>86.2</td>
      <td>7.5</td>
      <td>10.3</td>
      <td>60211.71</td>
      <td>34688.16</td>
      <td>26852.67</td>
      <td>43865.64</td>
      <td>49584</td>
      <td>98.6</td>
      <td>79.1</td>
      <td>17.9</td>
      <td>0.4</td>
      <td>0.6</td>
      <td>0.0</td>
      <td>0.7</td>
      <td>1.4</td>
      <td>2.0</td>
      <td>18288</td>
      <td>1475</td>
      <td>964</td>
      <td>16</td>
      <td>1099.0</td>
      <td>5028</td>
      <td>1519</td>
      <td>92</td>
      <td>2305.0</td>
      <td>29.54</td>
      <td>26.07</td>
      <td>33.40</td>
      <td>15.81</td>
      <td>14.16</td>
      <td>17.64</td>
      <td>28.75</td>
      <td>17.51</td>
      <td>41.25</td>
      <td>11.89</td>
      <td>20.30</td>
      <td>2.52</td>
      <td>14.02</td>
      <td>21.96</td>
      <td>5.19</td>
      <td>83.42</td>
      <td>85.68</td>
      <td>105.11</td>
      <td>105.11</td>
      <td>87.59</td>
      <td>87.39</td>
      <td>87.39</td>
      <td>85.68</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1001</td>
      <td>2010</td>
      <td>6.21</td>
      <td>5.57</td>
      <td>6.78</td>
      <td>5.1</td>
      <td>4.6</td>
      <td>5.6</td>
      <td>71.3</td>
      <td>67.0</td>
      <td>78.15</td>
      <td>59.71</td>
      <td>85.8</td>
      <td>7.5</td>
      <td>10.6</td>
      <td>61775.80</td>
      <td>34740.84</td>
      <td>27391.08</td>
      <td>46155.24</td>
      <td>53155</td>
      <td>98.5</td>
      <td>79.1</td>
      <td>17.9</td>
      <td>0.3</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>0.6</td>
      <td>1.5</td>
      <td>2.3</td>
      <td>19718</td>
      <td>1569</td>
      <td>1009</td>
      <td>16</td>
      <td>1110.0</td>
      <td>5472</td>
      <td>1541</td>
      <td>113</td>
      <td>2377.0</td>
      <td>29.33</td>
      <td>25.94</td>
      <td>33.06</td>
      <td>16.92</td>
      <td>15.09</td>
      <td>18.93</td>
      <td>29.07</td>
      <td>17.82</td>
      <td>41.43</td>
      <td>11.57</td>
      <td>19.86</td>
      <td>2.45</td>
      <td>13.11</td>
      <td>21.28</td>
      <td>4.13</td>
      <td>85.92</td>
      <td>89.96</td>
      <td>105.28</td>
      <td>105.28</td>
      <td>89.26</td>
      <td>91.33</td>
      <td>91.33</td>
      <td>89.96</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1001</td>
      <td>2011</td>
      <td>7.55</td>
      <td>8.13</td>
      <td>7.03</td>
      <td>6.2</td>
      <td>6.3</td>
      <td>6.1</td>
      <td>70.2</td>
      <td>66.5</td>
      <td>77.62</td>
      <td>59.31</td>
      <td>85.7</td>
      <td>7.4</td>
      <td>10.9</td>
      <td>60366.88</td>
      <td>34564.32</td>
      <td>26727.68</td>
      <td>45333.12</td>
      <td>53944</td>
      <td>98.5</td>
      <td>78.9</td>
      <td>18.1</td>
      <td>0.2</td>
      <td>0.7</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>1.5</td>
      <td>2.4</td>
      <td>19998</td>
      <td>1695</td>
      <td>1060</td>
      <td>106</td>
      <td>1030.0</td>
      <td>5065</td>
      <td>1965</td>
      <td>246</td>
      <td>2299.0</td>
      <td>31.17</td>
      <td>26.97</td>
      <td>35.96</td>
      <td>16.18</td>
      <td>14.21</td>
      <td>18.42</td>
      <td>27.56</td>
      <td>17.74</td>
      <td>38.76</td>
      <td>10.72</td>
      <td>18.28</td>
      <td>2.09</td>
      <td>14.38</td>
      <td>22.80</td>
      <td>4.77</td>
      <td>88.43</td>
      <td>94.25</td>
      <td>105.45</td>
      <td>105.45</td>
      <td>90.93</td>
      <td>95.28</td>
      <td>95.28</td>
      <td>94.25</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1001</td>
      <td>2012</td>
      <td>8.60</td>
      <td>8.88</td>
      <td>8.29</td>
      <td>6.7</td>
      <td>6.4</td>
      <td>7.0</td>
      <td>70.6</td>
      <td>67.1</td>
      <td>76.31</td>
      <td>58.30</td>
      <td>85.7</td>
      <td>7.4</td>
      <td>11.6</td>
      <td>59150.30</td>
      <td>34327.70</td>
      <td>27967.50</td>
      <td>44276.10</td>
      <td>54590</td>
      <td>98.5</td>
      <td>78.9</td>
      <td>18.1</td>
      <td>0.3</td>
      <td>0.8</td>
      <td>0.0</td>
      <td>0.4</td>
      <td>1.5</td>
      <td>2.4</td>
      <td>19934</td>
      <td>1714</td>
      <td>938</td>
      <td>120</td>
      <td>1095.0</td>
      <td>4608</td>
      <td>1963</td>
      <td>284</td>
      <td>2644.0</td>
      <td>32.13</td>
      <td>28.59</td>
      <td>36.09</td>
      <td>16.09</td>
      <td>14.71</td>
      <td>17.63</td>
      <td>28.39</td>
      <td>17.79</td>
      <td>40.26</td>
      <td>9.02</td>
      <td>16.03</td>
      <td>1.19</td>
      <td>14.37</td>
      <td>22.88</td>
      <td>4.84</td>
      <td>90.93</td>
      <td>98.53</td>
      <td>105.61</td>
      <td>105.61</td>
      <td>92.60</td>
      <td>99.22</td>
      <td>99.22</td>
      <td>98.53</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("\n\n County data\n\n")

df_county.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>county_fips_code</th>
      <th>county_name</th>
      <th>state_name</th>
      <th>state_abbreviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1001</td>
      <td>Autauga County</td>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1003</td>
      <td>Baldwin County</td>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1005</td>
      <td>Barbour County</td>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1007</td>
      <td>Bibb County</td>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1009</td>
      <td>Blount County</td>
      <td>Alabama</td>
      <td>AL</td>
    </tr>
  </tbody>
</table>
</div>




```python
print("\n\n Variable metadata\n\n")

costs_dict
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>class</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>county_fips_code</td>
      <td>double</td>
      <td>Four- or five-digit number that uniquely ident...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>study_year</td>
      <td>double</td>
      <td>Year the data collection began for the market ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>unr_16</td>
      <td>double</td>
      <td>Unemployment rate of the population aged 16 ye...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>funr_16</td>
      <td>double</td>
      <td>Unemployment rate of the female population age...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>munr_16</td>
      <td>double</td>
      <td>Unemployment rate of the male population aged ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>56</th>
      <td>mc_toddler</td>
      <td>double</td>
      <td>Aggregated weekly, full-time median price char...</td>
    </tr>
    <tr>
      <th>57</th>
      <td>mc_preschool</td>
      <td>double</td>
      <td>Aggregated weekly, full-time median price char...</td>
    </tr>
    <tr>
      <th>58</th>
      <td>mfcc_infant</td>
      <td>double</td>
      <td>Aggregated weekly, full-time median price char...</td>
    </tr>
    <tr>
      <th>59</th>
      <td>mfcc_toddler</td>
      <td>double</td>
      <td>Aggregated weekly, full-time median price char...</td>
    </tr>
    <tr>
      <th>60</th>
      <td>mfcc_preschool</td>
      <td>double</td>
      <td>Aggregated weekly, full-time median price char...</td>
    </tr>
  </tbody>
</table>
<p>61 rows × 3 columns</p>
</div>




```python
#Merging county data

df_costs = pd.merge(df_costs, df_county[['county_fips_code', 'county_name', 'state_name']], left_on = 'county_fips_code', right_on = 'county_fips_code')

df_costs['county_name'] = df_costs['county_name'].astype('category')
df_costs['state_name'] = df_costs['state_name'].astype('category')
```

## Data cleaning + EDA

Four steps implemented for data cleaning:

1. Duplicated and irrelevant information
2. Fix structural errors
3. Remove outliers
4. Fix missing data

### Duplicated and irrelevant information


```python
#1. Duplicated and irrelevant information
"""
In childcare data we have that each row is a study with a county code and study year,
so, a row can't be duplicated exactly in all columns (it's very rare).

In County data, we know that "fips" code is a unique code, so we can remove duplicated values from it.
"""

print("Rows before cleaning duplicates:\n childcare_costs:", df_costs.shape[0], "\n county_data:",df_county.shape[0])

df_costs = df_costs.drop_duplicates()
df_county = df_county.drop_duplicates(subset='county_fips_code')

print("Rows after cleaning duplicates:\n childcare_costs:", df_costs.shape[0], "\n county_data:",df_county.shape[0])
```

    Rows before cleaning duplicates:
     childcare_costs: 34567 
     county_data: 3144
    Rows after cleaning duplicates:
     childcare_costs: 34567 
     county_data: 3144
    

We can se no changes in rows, so there wasn't duplicated information!

### Fix structural errors

First let's check the data stats, looking for errors.


```python
df_costs.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>county_fips_code</th>
      <th>study_year</th>
      <th>unr_16</th>
      <th>funr_16</th>
      <th>munr_16</th>
      <th>unr_20to64</th>
      <th>funr_20to64</th>
      <th>munr_20to64</th>
      <th>flfpr_20to64</th>
      <th>flfpr_20to64_under6</th>
      <th>flfpr_20to64_6to17</th>
      <th>flfpr_20to64_under6_6to17</th>
      <th>mlfpr_20to64</th>
      <th>pr_f</th>
      <th>pr_p</th>
      <th>mhi_2018</th>
      <th>me_2018</th>
      <th>fme_2018</th>
      <th>mme_2018</th>
      <th>total_pop</th>
      <th>one_race</th>
      <th>one_race_w</th>
      <th>one_race_b</th>
      <th>one_race_i</th>
      <th>one_race_a</th>
      <th>one_race_h</th>
      <th>one_race_other</th>
      <th>two_races</th>
      <th>hispanic</th>
      <th>households</th>
      <th>h_under6_both_work</th>
      <th>h_under6_f_work</th>
      <th>h_under6_m_work</th>
      <th>h_under6_single_m</th>
      <th>h_6to17_both_work</th>
      <th>h_6to17_fwork</th>
      <th>h_6to17_mwork</th>
      <th>h_6to17_single_m</th>
      <th>emp_m</th>
      <th>memp_m</th>
      <th>femp_m</th>
      <th>emp_service</th>
      <th>memp_service</th>
      <th>femp_service</th>
      <th>emp_sales</th>
      <th>memp_sales</th>
      <th>femp_sales</th>
      <th>emp_n</th>
      <th>memp_n</th>
      <th>femp_n</th>
      <th>emp_p</th>
      <th>memp_p</th>
      <th>femp_p</th>
      <th>mcsa</th>
      <th>mfccsa</th>
      <th>mc_infant</th>
      <th>mc_toddler</th>
      <th>mc_preschool</th>
      <th>mfcc_infant</th>
      <th>mfcc_toddler</th>
      <th>mfcc_preschool</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.00000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>3.456700e+04</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>3.456700e+04</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34565.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34565.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>34567.000000</td>
      <td>23593.000000</td>
      <td>23383.000000</td>
      <td>23593.000000</td>
      <td>23593.000000</td>
      <td>23593.000000</td>
      <td>23383.000000</td>
      <td>23383.000000</td>
      <td>23383.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>30388.132786</td>
      <td>2012.999711</td>
      <td>7.465902</td>
      <td>7.02902</td>
      <td>7.860291</td>
      <td>6.900073</td>
      <td>6.482007</td>
      <td>7.275457</td>
      <td>70.086125</td>
      <td>68.821409</td>
      <td>78.824106</td>
      <td>66.940759</td>
      <td>78.637814</td>
      <td>11.739125</td>
      <td>16.038131</td>
      <td>50446.777363</td>
      <td>29404.295539</td>
      <td>23475.169710</td>
      <td>35997.433918</td>
      <td>9.914222e+04</td>
      <td>97.922863</td>
      <td>83.636247</td>
      <td>8.971213</td>
      <td>1.895634</td>
      <td>1.216209</td>
      <td>0.082101</td>
      <td>2.117914</td>
      <td>2.077146</td>
      <td>8.401054</td>
      <td>3.686299e+04</td>
      <td>2791.610004</td>
      <td>1821.628808</td>
      <td>141.109064</td>
      <td>1992.249964</td>
      <td>6480.568548</td>
      <td>2874.374374</td>
      <td>390.113056</td>
      <td>3924.511558</td>
      <td>30.720191</td>
      <td>26.495400</td>
      <td>35.509491</td>
      <td>17.909716</td>
      <td>13.122111</td>
      <td>23.424303</td>
      <td>22.244044</td>
      <td>13.660359</td>
      <td>32.062036</td>
      <td>13.076182</td>
      <td>23.265887</td>
      <td>1.369745</td>
      <td>16.049861</td>
      <td>23.456238</td>
      <td>7.634403</td>
      <td>101.234253</td>
      <td>92.523582</td>
      <td>146.051770</td>
      <td>130.482768</td>
      <td>122.232852</td>
      <td>113.421657</td>
      <td>106.759749</td>
      <td>104.189510</td>
    </tr>
    <tr>
      <th>std</th>
      <td>15161.015383</td>
      <td>3.162232</td>
      <td>3.538619</td>
      <td>3.56342</td>
      <td>4.037657</td>
      <td>3.446199</td>
      <td>3.477956</td>
      <td>3.990758</td>
      <td>7.696499</td>
      <td>11.758088</td>
      <td>8.529813</td>
      <td>13.285061</td>
      <td>10.802983</td>
      <td>5.681003</td>
      <td>6.511816</td>
      <td>13279.833788</td>
      <td>5715.192737</td>
      <td>4903.836942</td>
      <td>7643.855532</td>
      <td>3.177786e+05</td>
      <td>1.952523</td>
      <td>16.683515</td>
      <td>14.510260</td>
      <td>7.517786</td>
      <td>2.633768</td>
      <td>0.593602</td>
      <td>3.779858</td>
      <td>1.952520</td>
      <td>13.284933</td>
      <td>1.116470e+05</td>
      <td>8709.085062</td>
      <td>6740.389344</td>
      <td>472.557239</td>
      <td>7009.934446</td>
      <td>19294.344762</td>
      <td>10901.264277</td>
      <td>1187.096922</td>
      <td>13966.245194</td>
      <td>6.503783</td>
      <td>8.013214</td>
      <td>6.073988</td>
      <td>3.602868</td>
      <td>4.476415</td>
      <td>4.696024</td>
      <td>3.352621</td>
      <td>3.761803</td>
      <td>4.480591</td>
      <td>4.257660</td>
      <td>6.763236</td>
      <td>1.430796</td>
      <td>5.935141</td>
      <td>7.956927</td>
      <td>4.501044</td>
      <td>34.552888</td>
      <td>27.669904</td>
      <td>53.698566</td>
      <td>43.775370</td>
      <td>38.538323</td>
      <td>32.819372</td>
      <td>29.982431</td>
      <td>28.961701</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1001.000000</td>
      <td>2008.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>33.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.200000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>19841.920000</td>
      <td>4947.040000</td>
      <td>5019.300000</td>
      <td>4238.080000</td>
      <td>4.100000e+01</td>
      <td>59.500000</td>
      <td>3.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.200000e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>18.980000</td>
      <td>22.000000</td>
      <td>27.730000</td>
      <td>21.540000</td>
      <td>21.540000</td>
      <td>43.080000</td>
      <td>43.080000</td>
      <td>40.030000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>18177.000000</td>
      <td>2010.000000</td>
      <td>5.100000</td>
      <td>4.64000</td>
      <td>5.200000</td>
      <td>4.600000</td>
      <td>4.200000</td>
      <td>4.700000</td>
      <td>65.100000</td>
      <td>62.600000</td>
      <td>74.400000</td>
      <td>59.800000</td>
      <td>74.200000</td>
      <td>7.700000</td>
      <td>11.400000</td>
      <td>41839.215000</td>
      <td>25955.055000</td>
      <td>20613.100000</td>
      <td>31447.505000</td>
      <td>1.101450e+04</td>
      <td>97.500000</td>
      <td>77.000000</td>
      <td>0.600000</td>
      <td>0.200000</td>
      <td>0.200000</td>
      <td>0.000000</td>
      <td>0.300000</td>
      <td>1.100000</td>
      <td>1.700000</td>
      <td>4.236000e+03</td>
      <td>260.000000</td>
      <td>146.000000</td>
      <td>5.000000</td>
      <td>176.000000</td>
      <td>685.000000</td>
      <td>247.000000</td>
      <td>39.000000</td>
      <td>351.000000</td>
      <td>26.340000</td>
      <td>20.945000</td>
      <td>31.560000</td>
      <td>15.590000</td>
      <td>10.270000</td>
      <td>20.410000</td>
      <td>20.150000</td>
      <td>11.210000</td>
      <td>29.390000</td>
      <td>10.200000</td>
      <td>18.680000</td>
      <td>0.590000</td>
      <td>11.640000</td>
      <td>17.610000</td>
      <td>4.410000</td>
      <td>78.650000</td>
      <td>75.000000</td>
      <td>108.750000</td>
      <td>100.000000</td>
      <td>95.880000</td>
      <td>90.000000</td>
      <td>85.085000</td>
      <td>84.255000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>29177.000000</td>
      <td>2013.000000</td>
      <td>7.050000</td>
      <td>6.59000</td>
      <td>7.390000</td>
      <td>6.500000</td>
      <td>6.000000</td>
      <td>6.800000</td>
      <td>70.600000</td>
      <td>69.600000</td>
      <td>79.450000</td>
      <td>67.660000</td>
      <td>81.000000</td>
      <td>10.900000</td>
      <td>15.200000</td>
      <td>48505.600000</td>
      <td>28653.920000</td>
      <td>22854.700000</td>
      <td>35103.000000</td>
      <td>2.571100e+04</td>
      <td>98.400000</td>
      <td>90.200000</td>
      <td>2.100000</td>
      <td>0.300000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>0.800000</td>
      <td>1.600000</td>
      <td>3.400000</td>
      <td>9.814000e+03</td>
      <td>647.000000</td>
      <td>387.000000</td>
      <td>31.000000</td>
      <td>489.000000</td>
      <td>1662.000000</td>
      <td>620.000000</td>
      <td>116.000000</td>
      <td>907.000000</td>
      <td>29.690000</td>
      <td>25.190000</td>
      <td>34.940000</td>
      <td>17.580000</td>
      <td>12.840000</td>
      <td>23.090000</td>
      <td>22.410000</td>
      <td>13.680000</td>
      <td>32.170000</td>
      <td>12.560000</td>
      <td>22.700000</td>
      <td>0.990000</td>
      <td>15.590000</td>
      <td>23.400000</td>
      <td>6.630000</td>
      <td>96.530000</td>
      <td>88.180000</td>
      <td>134.500000</td>
      <td>120.990000</td>
      <td>113.990000</td>
      <td>106.000000</td>
      <td>100.250000</td>
      <td>99.650000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>45081.000000</td>
      <td>2016.000000</td>
      <td>9.350000</td>
      <td>8.88000</td>
      <td>9.920000</td>
      <td>8.700000</td>
      <td>8.250000</td>
      <td>9.200000</td>
      <td>75.500000</td>
      <td>76.100000</td>
      <td>84.225000</td>
      <td>75.200000</td>
      <td>85.900000</td>
      <td>14.600000</td>
      <td>19.500000</td>
      <td>56201.450000</td>
      <td>32005.820000</td>
      <td>25634.665000</td>
      <td>39503.335000</td>
      <td>6.670350e+04</td>
      <td>98.900000</td>
      <td>95.600000</td>
      <td>10.200000</td>
      <td>0.800000</td>
      <td>1.200000</td>
      <td>0.100000</td>
      <td>2.200000</td>
      <td>2.500000</td>
      <td>8.400000</td>
      <td>2.549350e+04</td>
      <td>1786.000000</td>
      <td>1099.000000</td>
      <td>103.000000</td>
      <td>1298.000000</td>
      <td>4469.500000</td>
      <td>1767.000000</td>
      <td>306.000000</td>
      <td>2421.000000</td>
      <td>34.100000</td>
      <td>30.720000</td>
      <td>38.870000</td>
      <td>19.780000</td>
      <td>15.450000</td>
      <td>26.010000</td>
      <td>24.540000</td>
      <td>16.240000</td>
      <td>34.800000</td>
      <td>15.340000</td>
      <td>27.210000</td>
      <td>1.650000</td>
      <td>19.830000</td>
      <td>28.930000</td>
      <td>9.920000</td>
      <td>119.380000</td>
      <td>107.500000</td>
      <td>166.330000</td>
      <td>148.710000</td>
      <td>139.300000</td>
      <td>129.315000</td>
      <td>124.950000</td>
      <td>120.200000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>56045.000000</td>
      <td>2018.000000</td>
      <td>36.110000</td>
      <td>38.24000</td>
      <td>39.740000</td>
      <td>33.900000</td>
      <td>44.500000</td>
      <td>45.500000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>52.100000</td>
      <td>55.100000</td>
      <td>136268.000000</td>
      <td>70565.000000</td>
      <td>68600.000000</td>
      <td>98580.000000</td>
      <td>1.010572e+07</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>87.400000</td>
      <td>95.700000</td>
      <td>52.200000</td>
      <td>35.300000</td>
      <td>58.800000</td>
      <td>40.500000</td>
      <td>99.200000</td>
      <td>3.306109e+06</td>
      <td>264439.000000</td>
      <td>246269.000000</td>
      <td>15396.000000</td>
      <td>220530.000000</td>
      <td>590730.000000</td>
      <td>391225.000000</td>
      <td>40499.000000</td>
      <td>436538.000000</td>
      <td>74.030000</td>
      <td>76.890000</td>
      <td>100.000000</td>
      <td>46.420000</td>
      <td>56.330000</td>
      <td>62.780000</td>
      <td>50.770000</td>
      <td>49.060000</td>
      <td>100.000000</td>
      <td>60.000000</td>
      <td>80.240000</td>
      <td>27.270000</td>
      <td>72.850000</td>
      <td>87.500000</td>
      <td>66.480000</td>
      <td>375.400000</td>
      <td>308.000000</td>
      <td>470.000000</td>
      <td>419.000000</td>
      <td>385.000000</td>
      <td>430.940000</td>
      <td>376.320000</td>
      <td>331.340000</td>
    </tr>
  </tbody>
</table>
</div>



Insights:

* The columns related to rates are numeric and have values between 0 and 100 (the normal range for a rate is 0%-100%).
* The columns related to money are non-negative and numeric.
* The population column contains non-zero, numeric values.

No structural transformations are necessary.

### Remove outliers - Univariate exploratory analysis

Removing outliers can be tricky, and it will depend on our business context. In this case, we don’t have a business expert to help us, so we must proceed carefully. let's check our outliers column by column with a widget:


```python
#Box and whiskers plot for checking outliers:

def boxplot(variable):
    plt.figure(figsize=(5,4))
    sns.boxplot(data=df_costs, x=variable, whis=3)
    plt.show()

# widget
variable_widget = widgets.Dropdown(options=df_costs.columns)

# update plot with another column
widgets.interactive(boxplot, variable=variable_widget)

```




![png](chilldcare_eda_files/univariate_boxplot.png)



For now, we have found some outliers (Q3 + 3*IQR data) in some columns. We can do 2 things:

1. Assume that all our data have no human errors and don’t eliminate outliers yet because we want to check the full behavior of our data.
2. Eliminate outliers based on a statistical formula: Preserve data between (Q1 - 3IQR) and (Q3 + 3IQR) for each column.


In this case, we will proceed with option 1 (for check all data behaviour) and not eliminate data yet.

### Fix missing data

Now we will handle our NaN values, firts let's check how many columns have NaNs:


```python
#Overview of null values:

# Check for missing values
null_data = round(df_costs.isna().sum().sort_values(ascending = False)/df_costs.shape[0]*100, 1)
plt.bar(null_data.index, null_data.values)
plt.title('Percentage of null values in columns')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

null_data[0:11]
```


    
![png](chilldcare_eda_files/chilldcare_eda_22_0.png)
    





    mfccsa               32.4
    mfcc_preschool       32.4
    mfcc_toddler         32.4
    mfcc_infant          32.4
    mcsa                 31.7
    mc_preschool         31.7
    mc_toddler           31.7
    mc_infant            31.7
    h_6to17_single_m      0.0
    h_under6_single_m     0.0
    femp_service          0.0
    dtype: float64



We have 8 columns with around 31% NaN values each one, that columns are related to median prices charged for childcare (thats the core of our analysis), so we will drop that data with NaNs.


```python
#Drop NaN data
shape = df_costs.shape[0]
df_costs = df_costs.dropna()
print("Rows eliminated:", shape - df_costs.shape[0])
```

    Rows eliminated: 11225
    

## Bivariate analysis

### Correlation

Correlation analysis in a nutshell:

If two features are closely or completely correlated (or inverse correlated), an increase in one results in an increase in the other (a decrease in one will increase the other in case of inverse correlation). This implies that both features carry very similar (or inverse) information, with little to no variation. This phenomenon is known as MultiColinearity, as both features essentially convey the same information. therefore we need eliminate redundant information from our data, for saving space in memory, faster training in models, avoiding bias in modeling, etc.


```python
#Correlation heatmap
plt.figure(figsize=(30,25))
sns.heatmap(round(df_costs.drop(['county_name', 'state_name'], axis = 1).corr(),2), annot=True, cmap='RdYlGn', linewidths=0.2)
plt.xticks(rotation=90) # Rotates X-Axis Ticks by 90-degrees
plt.yticks(rotation=0) # Rotates Y-Axis Ticks
plt.show()
```


    
![png](chilldcare_eda_files/chilldcare_eda_26_0.png)
    


### Interesting relationships:

- % of civilians in service jobs has an inverse relationship with mean household income (-35%) and labor force participation (-42%), and a positive relationship with the poverty rate (+33%).
- % of civilians in sales/office-related jobs has a proportional relationship with mean household income (+17%), but a +28% relationship with males and +2% for females.

- Childcare costs are highly positively correlated with:
    - median incomes (+50%)
    - Labor force participation (+18% for males, +26% for females)
    - % of civilians in managment, science, art and business occupations (+38%)
    - % of male civilians in sales and office jobs (+23%), but for females, it's -10%
    
* Childcare costs are highly negatively correlated with:
    - Poverty rate for families (-31%)
    - % of civilians employed in natural resources, construction, and mantenance occupations (-25%), (for males -28%, for females -1%).
    - % of civilians employed in transportation, production and material moving (-30%)

* The racial composition of a county (white, black, native) has a significant relationship with poverty, income, unemployment, and job type.

### To handle multicollinearity issues in regression/classification modeling, we can choose to:

- Filter out highly correlated variables to avoid multicollinearity issues (above 80% of positive or negative correlation).
- Use regularization methods (i.e., L1 or L2 regularization, Dropout, data augmentation, etc.)
- Use dimensionality reduction techniques (i.e., PCA, Factor Analysis, Linear Discriminant Analysis, etc.)

By now, we will not transform any data, that is for future analysis.

## Explore categorical vs numerical data

Now let's transform numeric data into categorical binned data, for EDA purposes, our main question is:

- How is the distribution behaviour with high, medium and low levels for each variable?


```python
#Create new categorical columns
descriptive = df_costs.describe()
def to_categorical(column, descriptive):
    bins = [-np.inf, descriptive.loc['25%', column], descriptive.loc['50%', column], descriptive.loc['75%', column], np.inf]
    if len(set(bins)) == len(bins): #Check if our bins have repeated values. 
        return pd.cut(
                df_costs[column], 
                bins=bins, 
                labels=['p0-p25', 'p25-p50', 'p50-p75', 'p75-p100'], 
                ordered=False,
                )
    else: #If our bins have repeated values, create bins by the range of our column
        return pd.cut(
                df_costs[column], 
                bins=3, 
                labels=['low', 'mid', 'high'], 
                ordered=False,
                )
        

numeric_options = ['unr_16', 'funr_16', 'munr_16',
       'unr_20to64', 'funr_20to64', 'munr_20to64', 'flfpr_20to64',
       'flfpr_20to64_under6', 'flfpr_20to64_6to17',
       'flfpr_20to64_under6_6to17', 'mlfpr_20to64', 'pr_f', 'pr_p', 'mhi_2018',
       'me_2018', 'fme_2018', 'mme_2018', 'total_pop', 'one_race',
       'one_race_w', 'one_race_b', 'one_race_i', 'one_race_a', 'one_race_h',
       'one_race_other', 'two_races', 'hispanic', 'households',
       'h_under6_both_work', 'h_under6_f_work', 'h_under6_m_work',
       'h_under6_single_m', 'h_6to17_both_work', 'h_6to17_fwork',
       'h_6to17_mwork', 'h_6to17_single_m', 'emp_m', 'memp_m', 'femp_m',
       'emp_service', 'memp_service', 'femp_service', 'emp_sales',
       'memp_sales', 'femp_sales', 'emp_n', 'memp_n', 'femp_n', 'emp_p',
       'memp_p', 'femp_p', 'mcsa', 'mfccsa', 'mc_infant', 'mc_toddler',
       'mc_preschool', 'mfcc_infant', 'mfcc_toddler', 'mfcc_preschool']

for column in numeric_options:
    df_costs['cat_' + column] = to_categorical(column=column, descriptive=descriptive)

category_options = ['cat_' + i for i in numeric_options]
```

### Multivariate visualization

The following plots can show us the behaviour of this combination of variables:

- Numerical variable (for histogram)
- Categorical variable (for x-axis bins)
- Optional categorical variable (for hue in each x-axis bins)

If you just want to plot the numerical variable against one categorical variable, you simply need to set variable3 to the "None" option.

This allows us to understand interactions between variables in a graphical way!

### Scatter plot

Let's check bivariate relationships with a scatter plot widget:


```python
#Box and whiskers plot for checking outliers:

def scatter(variable1, variable2, variable3):
    plt.figure(figsize=(5,4))
    if variable3 != 'None':
        sns.jointplot(data=df_costs, x=variable1, y=variable2, hue = variable3)
        print("", costs_dict[costs_dict['variable'] == variable1]['description'].values[0], "\n\n VS\n",
               costs_dict[costs_dict['variable'] == variable2.replace('cat_','')]['description'].values[0],
              "\n\n BY:\n", 
              costs_dict[costs_dict['variable'] == variable3.replace('cat_','')]['description'].values[0])
        plt.show()
    else:
        sns.jointplot(data=df_costs, x=variable1, y=variable2, kind="reg")
        print("", costs_dict[costs_dict['variable'] == variable1]['description'].values[0], "\n VS\n",
               costs_dict[costs_dict['variable'] == variable2.replace('cat_','')]['description'].values[0])
        plt.show()

# widget
variable1_widget = widgets.Dropdown(options=numeric_options, value = 'unr_16')
variable2_widget = widgets.Dropdown(options=numeric_options, value = 'mhi_2018')
variable3_widget = widgets.Dropdown(options=['None'] + category_options, value = 'cat_emp_m')

# update plot with another column
widgets.interactive(scatter, variable1=variable1_widget, variable2=variable2_widget, variable3=variable3_widget)
```




![png](chilldcare_eda_files/jointplot.png)




```python
#Create violin plot for checking categorical behaviour

def violinplot(variable1, variable2, variable3):
    plt.figure(figsize=(10,6))
    if variable3 != 'None':
        sns.violinplot(data=df_costs, y=variable1, x=variable2, hue = variable3, split=False, inner="quart")
        print("", costs_dict[costs_dict['variable'] == variable1]['description'].values[0], "\n\n VS\n",
               costs_dict[costs_dict['variable'] == variable2.replace('cat_','')]['description'].values[0],
              "\n\n BY:\n", 
              costs_dict[costs_dict['variable'] == variable3.replace('cat_','')]['description'].values[0])
        plt.show()
    else:
        sns.violinplot(data=df_costs, y=variable1, x=variable2, inner="quart")
        print("", costs_dict[costs_dict['variable'] == variable1]['description'].values[0], "\n VS\n",
               costs_dict[costs_dict['variable'] == variable2.replace('cat_','')]['description'].values[0])
        plt.show()

# widget
variable1_widget = widgets.Dropdown(options=numeric_options, value = 'mcsa')
variable2_widget = widgets.Dropdown(options=category_options, value = 'cat_hispanic')
variable3_widget = widgets.Dropdown(options=['None'] + category_options, value = 'cat_pr_f')

# update plot with another column
widgets.interactive(violinplot, variable1=variable1_widget, variable2=variable2_widget, variable3=variable3_widget)
```




![png](chilldcare_eda_files/Violinplot.png)



# Conclusions

## What we did?

* Data check and cleaning: Cleaning the data have improved the quality of our dataset by checking our data quality and handling missing values. This is crucial for obtaining accurate and reliable results in any subsequent analysis.

* Univariate analysis: We had numeric columns, so we did a dispersion analysis, for checking outliers and possible data errors.

* Bivariate analysis: Looking for relationships between variables we did the folowing analysis:

    - Creation of Categorical Columns Based on Percentiles of Numerical Columns: This would have allowed for a more intuitive interpretation of the numerical variables by grouping them into categories based on their distribution. This can also help reveal patterns that are not apparent when considering the raw numerical values.

    - Correlation Analysis: We found strongly correlated variables that could indicate redundancy in our dataset, while weak correlations could suggest more complex relationships that could be further explored.

    - Violin Plots with Categorical HUE: The violin plots have provided a detailed view of the distribution of our variables, with the added advantage of the categorical hue showing how these distributions differ between categories. This revealed key differences between groups.

    - Joinplots with Categorical HUE: The joinplots have provided a joint view of two variables, allowing us to see both the relationship between them (through the scatter plot) and their individual distributions (through the histograms). The categorical hue adds another dimension to this analysis, allowing us to see how these relationships may change between different categories.


## Next steps?

- We can perform an in-depth analysis of the relationships between variables by conducting T-tests, ANOVAs, simple regressions, etc.
- We can check for homoscedasticity in our variables with the Breusch-Pagan test, White test, or by creating a plot of Fitted Values vs. Residuals. 
- We can also determine the type of distribution followed by our data using the Kolmogorov-Smirnov test, Chi-Square test, Jarque-Bera Tests, etc.

With this data, we can carry out descriptive-predictive analyses on any of the variables in our dataset. We just need to set an objective like:

- "Predict the median household income"
- "Clustering counties by selecting variables"
- "Predict the childcare costs for a county"

The steps required for each analysis will be different, and our data transformation too! But that's not our current objective.

Thanks for reading!
