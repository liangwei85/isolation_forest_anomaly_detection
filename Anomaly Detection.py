#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
# Set the max_columns option to None (which means "unlimited")
pd.set_option('display.max_columns', None)
import warnings
# Option A: Ignore the warning globally for the current session
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

import numpy as np
# --- Patch for NumPy 2.0 compatibility ---
if not hasattr(np, 'VisibleDeprecationWarning'):
    np.VisibleDeprecationWarning = UserWarning

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import sweetviz as sv

from itertools import combinations
from datetime import datetime
import time

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import IsolationForest



# # Section 2: Input Data
# In semiconductor industry, a wafer goes through fabrication process that involves various tools, and a recipe defines how the wafer is processed by a tool. A wafer lot is a batch of multiple wafers that belong to the same technology and product.
# The sensor values of a tool are recorded for each wafer run, and this data is important for engineers to determine if the tool is working properly or not. Therefore, an FDC (fault detection and classification) system is necessary to detect any tool sensor anomalies to prevent further wafer scraps.
# The CSV file contains the wafer run information (e.g. TimeStamp, ToolName), and the columns with randomized characters denote the sensors. 
# 

# In[2]:


df_initial = pd.read_csv("Tool_Sensor_Data.csv")


# In[3]:


df_initial.shape


# In[4]:


df_initial.head()


# # Section 3: Data Processing and Exploratory Data Analysis (EDA)
# - You are required to apply data processing (e.g. data cleaning, filtering) and perform EDA to understand the underlying characteristics of the data. 
# - The results should be clearly visualized and are presentable (graphs, tables, charts, etc.). 
# - You may also optionally create a dashboard/app depicting the analysis results.
# 

# In[5]:


# Analyse Data 
report_initial = sv.analyze(df_initial)


# In[6]:


report_initial.show_notebook()


# # Section 4: Database Design and AI Modelling
# ## B.	Anomaly Detection Modelling and Pipeline Design (Applicable for Data Scientist/AI Engineer Candidate(s) Only)
# - The task is to create a data science pipeline that performs anomaly detection of tool sensors. 
# - The solution should be able to predict whether a wafer run is anomalous or not. 
# - There are no labels (normal/anomalous) associated to the sensor data. A typical example of the pipeline should include the following processes:
#     - 1.	Data preprocessing
#     - 2.	Feature engineering
#     - 3.	Feature selection
#     - 4.	Hyperparameter tuning
#     - 5.	Model training
#     - 6.	Model prediction
#     - 7.	Anomaly detection
#     - 8.	Model explainability (optional)

# ## 1.0 Data preprocessing

# ### 1.1 Remove the rows with empty readings. (These rows are initializing and ending logs)

# In[7]:


df_initial = df_initial[~df_initial['EventType'].str.contains('StartOfRun|EndOfRun', case=False, na=False)]
print(f"Original number of columns: {df_initial.shape[1]}")


# In[8]:


df_initial.shape


# ### 1.2 Remove Columns with 100% missing variables.

# In[9]:


df_clean = df_initial.dropna(axis=1, how='all')
print(f"Original number of columns: {df_clean.shape[1]}")


# ### 1.3 Remove Low Cardinalinity Variables (Low predictive power)

# In[10]:


# 1. Identify columns with only 1 unique value
single_val_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1]

# 2. Drop them
df_clean = df_clean.drop(columns=single_val_cols)


print(f"Number of Dropped columns: {len(single_val_cols)}\n")
print(f"Dropped columns: {single_val_cols}\n")
print(f"Original number of columns: {df_clean.shape[1]}\n")


# ### 1.4 Remove Highly Correlated Variables

# In[11]:


def remove_perfectly_correlated_features(df):
    """
    Removes one feature from every pair of perfectly correlated features 
    (where correlation is exactly 1.0 or -1.0) in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: A new DataFrame with one redundant feature removed from 
                      each perfectly correlated pair.
    """
    # 1. Select only numerical columns for correlation calculation
    numerical_df = df.select_dtypes(include=['number'])

    # 2. Calculate the correlation matrix
    corr_matrix = numerical_df.corr().abs()

    # 3. Create a mask to identify pairs to check
    # We use triu (upper triangle) and k=1 to exclude the diagonal (A vs A) 
    # and avoid checking pairs twice (A vs B and B vs A)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # 4. Find columns where the correlation is exactly 1.0
    to_drop = [column for column in upper.columns if any(upper[column] >= 0.90)]

    # 5. Drop the identified redundant columns from the numerical subset
    cleaned_df = df.drop(columns=to_drop)
    
    # Report the findings
    print(f"Original number of columns: {df.shape[1]}")
    print(f"Number of perfectly correlated features identified and dropped: {len(to_drop)}")
    if to_drop:
        print(f"Dropped columns: {to_drop}")

    return cleaned_df

df_clean = remove_perfectly_correlated_features(df_clean)

print(f"Original number of columns: {df_clean.shape[1]}\n")


# ### 1.5 Impute For categorical as missing Values in Sensors might mean something

# In[12]:


# Filter out Numerical Cols
numerical_cols = list(df_clean.select_dtypes(include=['number']).iloc[:,1:].columns)

all_columns = df_clean.columns
mask = ~all_columns.isin(numerical_cols)

# Reverse filter the numerical cols
categorical_cols = list(df_clean.loc[:, mask].columns)


# In[13]:


imputation_word = 'Missing'

# Impute the missing values in the specified columns
for cat_cols in categorical_cols[3:]:
    df_clean[cat_cols] = df_clean[cat_cols].fillna(imputation_word)


# In[14]:


df_clean.shape


# ### 1.6 Standardize Value for all Sensors

# In[15]:


# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        # Apply the StandardScaler to the specified numerical columns
        ('numerical_scaling', StandardScaler(), numerical_cols)
    ],
    remainder='drop' # Explicitly drops any columns not listed above
)

# Apply the transformation to the data
scaled_array = preprocessor.fit_transform(df_clean.loc[:,numerical_cols])

# Create the list of final column names in the correct order
final_cols = numerical_cols  

# Convert the resulting NumPy array back to a DataFrame
df_scaled_numeric = pd.DataFrame(scaled_array, columns=final_cols).reset_index(drop=True)

df_clean_cat_cols = df_clean.loc[:,categorical_cols].reset_index(drop=True)

df_clean = pd.concat([df_clean_cat_cols,df_scaled_numeric], axis=1)

# Reset index to prevent join problems
df_clean.reset_index(drop=True, inplace=True)


# ### 1.7 One Hot Encoding due to low Cardinality

# In[16]:


categorical_cols1 = categorical_cols[3:]


# In[17]:


# Perform One-Hot Encoding with the 'drop_first=True' parameter
df_clean = pd.get_dummies(df_clean, columns=categorical_cols1, dtype=int)


# ### 1.8 Convert String to Timestamp

# In[18]:


# Convert Timestamp str to timestamp
df_clean['TimeStamp'] = pd.to_datetime(df_clean['TimeStamp'])


# ### 1.9 "Run" column is record column and should treated as categorical 

# In[19]:


df_clean["Run"] = df_clean["Run"].astype(str)


# ## 2.0 Feature Engineering

# In[20]:


# Analyse Data 
report_clean = sv.analyze(df_clean)


# In[21]:


report_clean.show_notebook()


# ### 2.1 Analysis of run sensor reading frequency and duration per run

# #### 2.1.1 Each Run has an average of 59 readings and about 4 specific time stamps. Each time stamp is exactly a minute

# In[22]:


time_per_run = df_clean.pivot_table(index='Run', 
                                    values='TimeStamp', 
                                    aggfunc=['min','max','count','nunique'])

# Collapse the double headers into one string
# This turns ('Sales', 'sum') into "Sales_sum"
time_per_run.columns = ['_'.join(col).strip() for col in time_per_run.columns.values]

# Flatten it
time_per_run.reset_index(inplace=True)

# Calculate the difference (creates a 'timedelta' object)
time_per_run['total_run_sec'] = time_per_run['max_TimeStamp'] - time_per_run['min_TimeStamp']
# time_per_run['start_to_run_sec'] = time_per_run['min_TimeStamp'] - time_per_run['RunStartTime']
time_per_run['total_run_sec'] = time_per_run['total_run_sec'].dt.total_seconds() 


# In[23]:


### Summary Statistics of the runs
time_per_run.describe(include='all')


# ### 2.2 Feature Engineering -  Time Series functions & Lag movements

# In[25]:


# In[24] - Updated for Group-wise Application
LAG_SIZE = 1
ROLLING_WINDOW = 5

for col in numerical_cols:
    print(f"Engineering features for: {col}")
    
    # Apply time-series features grouped by the 'Run' column
    grouped = df_clean.groupby('Run')[col]
    
    # 3a. ABSOLUTE & PERCENTAGE DIFFERENCE (Rate of Change)
    df_clean[f'{col}_Diff_Abs'] = grouped.diff(LAG_SIZE)
    df_clean[f'{col}_Diff_Pct'] = grouped.pct_change(LAG_SIZE)
    
    # 3b. LAG FEATURE
    df_clean[f'{col}_Lag_{LAG_SIZE}'] = grouped.shift(LAG_SIZE)

    # 3c. ROLLING WINDOW STATISTICS
    prefix = f'{col}_W{ROLLING_WINDOW}' 
    
    # Rolling Mean - use .transform() to return results aligned to the original index
    df_clean[f'{prefix}_Mean'] = grouped.rolling(
        window=ROLLING_WINDOW, min_periods=1
    ).mean().reset_index(level=0, drop=True) # Important step to align index
    
    # Rolling Standard Deviation
    df_clean[f'{prefix}_Std'] = grouped.rolling(
        window=ROLLING_WINDOW, min_periods=1
    ).std().reset_index(level=0, drop=True)

    # Deviation from Mean
    df_clean[f'{prefix}_Deviation'] = df_clean[col] - df_clean[f'{prefix}_Mean']


# ### 2.3 Feature Engineering - Temporal Anomalies

# In[26]:


def encode_cycle(df, col_name, max_val):
    """Encodes a cyclical feature using sine and cosine transforms."""
    # Extract the component first (e.g., minute, hour, dayofweek)
    if col_name == 'Hour':
        component = df['TimeStamp'].dt.hour
    elif col_name == 'Day_of_Week':
        component = df['TimeStamp'].dt.dayofweek
    else: # Default case for flexibility
        raise ValueError("Unsupported temporal component.")
        
    df[col_name + '_sin'] = np.sin(2 * np.pi * component / max_val)
    df[col_name + '_cos'] = np.cos(2 * np.pi * component / max_val)
    return df

# Encode Hour of Day (max_val = 24)
df_clean = encode_cycle(df_clean, 'Hour', 24)

# Encode Day of Week (max_val = 7)
df_clean = encode_cycle(df_clean, 'Day_of_Week', 7)



# #### 2.3.1 Impute data created by features due to lag feature

# In[28]:


# In[26] - Simplified Imputation for *newly created* features
# Identify only the newly engineered columns that contain NaN
new_feature_cols = [col for col in df_clean.columns if '_Diff_' in col or '_Lag_' in col or '_W' in col]

# Calculate the median for only the newly created columns
medians_new_features = df_clean[new_feature_cols].median()

# Fill NaNs only in the new feature columns using their respective medians
df_clean[new_feature_cols] = df_clean[new_feature_cols].fillna(medians_new_features)


# #### 2.3.1 Remove highly correlated variables

# In[29]:


df_clean = remove_perfectly_correlated_features(df_clean)

print(f"Original number of columns: {df_clean.shape[1]}\n")


# ## 3.0 Feature Selection 

# - These methods focus on reducing redundancy or dimensionality in your feature space, 
# - Which helps Isolation Forest work faster and often improves detection quality by removing noise.

# ### 3.1 Variance Thresholding
# 
# - Method: Remove features whose variance is lower than a specified threshold.2 Low-variance features are nearly constant and provide little discriminatory information.3Why it helps IF: Constant features don't help in isolating anomalies and can just add noise, especially in high-dimensional data.Action: Calculate the variance of all features. Drop any feature $X_i$ where $\text{Var}(X_i) < \epsilon$.

# In[30]:


## Only extract numeric columns
df_numeric = df_clean.iloc[:,3:].reset_index(drop=True)

## Lower threshold as we want to preserve the outliers
threshold_value = 0.05
selector = VarianceThreshold(threshold=threshold_value)

# Fit the selector to the data and transform it (remove low-variance columns)
selected_features_array = selector.fit_transform(df_numeric)


# Reconstruct DataFrame and Identify Dropped Columns ---
feature_mask = selector.get_support() 

# Get the names of the selected columns
selected_column_names = df_numeric.columns[feature_mask]

# Reconstruct the DataFrame with only the selected columns
df_selected = pd.DataFrame(
    selected_features_array, 
    columns=selected_column_names, 
    index=df_numeric.index
)

# Combine with records again.
df_clean2 = pd.concat([df_clean.iloc[:,:3], df_selected], axis=1)



# ### 3.2 Feature Selection 

# In[31]:


# Select only numeric columns
FEATURE_CANDIDATES = df_clean2.columns.tolist()[3:] 


# In[32]:


# --- Custom Evaluation Metric (Required from previous step) ---
def evaluate_isolation_scores(df_subset, contamination=0.05):
    if df_subset.empty: return -np.inf
    X = StandardScaler().fit_transform(df_subset)
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42, n_jobs=-1)
    model.fit(X)
    scores = model.decision_function(X)
    contamination_count = int(len(scores) * contamination)
    median_score = np.median(scores)
    mean_anomaly_score = np.mean(np.sort(scores)[:contamination_count])
    return median_score - mean_anomaly_score


# In[33]:


# --- GREEDY FORWARD SELECTION ALGORITHM ---

best_score = -np.inf
current_features = []
available_features = FEATURE_CANDIDATES.copy()
improvement_threshold = 0.001 # Minimum score improvement required to add a feature

print(f"Starting Greedy Forward Selection with {len(available_features)} initial features...")
print("-" * 50)

while available_features:
    best_candidate = None
    best_gain = 0
    # Variable to hold the actual score of the best candidate in this iteration
    best_candidate_score = -np.inf
    
    # 1. Start timer for this iteration
    iteration_start_time = time.time()
    
    # 2. Iterate through ALL available features
    for candidate in available_features:
        # Create a test set by adding the candidate to the current best set
        test_set = current_features + [candidate]
        
        # Evaluate the new set
        score = evaluate_isolation_scores(df_clean2[test_set])
        gain = score - best_score
        
        # Check if this candidate is the best improvement so far
        if gain > best_gain:
            best_gain = gain
            best_candidate = candidate
            # Store the actual score of this best candidate
            best_candidate_score = score
            
    # 3. Decision Time: Should we keep the best candidate?
    if best_candidate and best_gain >= improvement_threshold:
        current_features.append(best_candidate)
        available_features.remove(best_candidate)
        
        # FIX: Assign the actual score of the new feature set, avoiding the -inf + inf error.
        best_score = best_candidate_score 
        
        elapsed = time.time() - iteration_start_time
        print(f"[{elapsed:.2f}s] ADDED: {best_candidate} | New Score: {best_score:.4f} | Gain: {best_gain:.4f}")
        
    else:
        # No feature improved the score by the threshold (or at all)
        print("\nStopping criterion met: No further feature significantly improved the score.")
        break
        
print("-" * 50)
print(f"FINAL OPTIMAL FEATURE SET ({len(current_features)} features):")
print(current_features)
print(f"Final Score Gap: {best_score:.4f}")


# ## 4.0 Hyperparameter Tuning

# In[34]:


RANDOM_STATE = 42
# Set your expected anomaly rate here
EXPECTED_CONTAMINATION_RATE = 0.01 # expect at least 99% of wafers to be usable for costs purposes

# Define feature columns
FEATURE_COLS = current_features
X = df_clean2[FEATURE_COLS].values 

N_SAMPLES_TOTAL = len(X)

print(f"Dataset loaded with {N_SAMPLES_TOTAL} samples.")
print(f"Model will be trained assuming a contamination rate of {EXPECTED_CONTAMINATION_RATE*100:.1f}%.")


# ## 5.0 Model Training  

# In[35]:


# The IsolationForest estimator.
# The contamination parameter is set based on the domain expectation.
if_model = IsolationForest(
    n_estimators=100,             # Default is usually a good start
    max_samples='auto',           # 'auto' is min(256, n_samples)
    contamination=EXPECTED_CONTAMINATION_RATE, 
    random_state=RANDOM_STATE,
    n_jobs=-1 # Use all processors for faster training
)

print("\nFitting Isolation Forest Model...")

# Fit the model to your unlabeled feature data
if_model.fit(X)

print("Model fitting complete.")


# ## 6.0 Model Prediction

# In[36]:


# Anomaly Scores (Lower score = more anomalous)
# This is the raw measure of how "isolated" a point is.
anomaly_scores = if_model.decision_function(X) 
df_clean2['Anomaly_Score'] = anomaly_scores

# Anomaly Predictions
# predict() returns: -1 for outliers/anomalies, 1 for inliers/normal points
anomaly_predictions = if_model.predict(X)
df_clean2['Anomaly_Label'] = anomaly_predictions


# ## 7.0 Anomaly Detection

# In[37]:


# Count the number of detected anomalies, which should closely match 
# the expected contamination rate times the total sample size.
n_anomalies_detected = (df_clean2['Anomaly_Label'] == -1).sum()
expected_anomalies = int(N_SAMPLES_TOTAL * EXPECTED_CONTAMINATION_RATE)


# In[38]:


# --- 4. Results and Analysis ---
print("\n--- Unsupervised Detection Results ---")
print(f"Total samples: {N_SAMPLES_TOTAL}")
print(f"Expected number of anomalies (based on contamination={EXPECTED_CONTAMINATION_RATE}): {expected_anomalies}")
print(f"Detected anomalies (records with Anomaly_Label == -1): {n_anomalies_detected}")

# Display the top 5 most anomalous records (lowest scores)
top_anomalies = df_clean2.sort_values(by='Anomaly_Score', ascending=True).head(5)

top_features = [extra_cols for extra_cols in current_features] + [extra_cols for extra_cols in ['Anomaly_Score','Anomaly_Label']]

top_anomalies.loc[:,top_features]


# In[39]:


## View Top Anomolous Data Looks like 
df_clean2.loc[:,top_features].sort_values(by='Anomaly_Score', ascending=True).head(20)


# In[40]:


# Analyse Data to get a feel of how anomoulous the data are
model_report_clean = sv.analyze(df_clean.loc[:, current_features])


# In[41]:


model_report_clean.show_notebook()


# ### 7.1 Visualisation of the Anomolous Points with Normal Data

# <span style="background-color:red; color:white; font-size:200%">
# ## Observable Notes </span> 
# <p>
# <span style="background-color:red; color:white">* Despite high movements in the sensor readings from the previous readings... Data Quality remained indifferent. </span> 
# <span style="background-color:red; color:white">* Which is weird. Like the readings don't matter to data quality despite large movements in the sensors</span> </p>

# In[42]:


## Load subset for Visualisation 
df = df_clean2.loc[:,top_features].sort_values(by='Anomaly_Score', ascending=True)

# 2. DEFINE THE 4 COLUMNS 
columns_to_plot = current_features


# 3. CREATE THE PAIRPLOT (SCATTER MATRIX)
fig = px.scatter_matrix(
    df,
    # Use the 'dimensions' argument to select the columns
    dimensions=columns_to_plot, 
    color='Anomaly_Label',
    color_continuous_scale="picnic",
    title="Pairplot (Scatter Matrix) of Selected Columns",
    height=1080,
    width=1080
)

# 4. OPTIONAL: Customize the plot appearance
# Hides the upper triangle for cleaner visualization and makes the diagonal visible
fig.update_traces(diagonal_visible=True, showupperhalf=False) 

# 5. SHOW THE PLOT
fig.show()


# ### 7.2 Visualisation of the Anomolous Points Only

# In[43]:


## Load subset for Visualisation 
df = df_clean2.loc[df_clean2['Anomaly_Label']==-1,top_features].sort_values(by='Anomaly_Score', ascending=True)

# 2. DEFINE THE 4 COLUMNS 
columns_to_plot = current_features


# 3. CREATE THE PAIRPLOT (SCATTER MATRIX)
fig = px.scatter_matrix(
    df,
    # Use the 'dimensions' argument to select the columns
    dimensions=columns_to_plot, 
    color='Anomaly_Label',
    color_discrete_sequence='Viridis',
    title="Pairplot (Scatter Matrix) of Selected Columns",
    height=1080,
    width=1080
)

# 4. OPTIONAL: Customize the plot appearance
# Hides the upper triangle for cleaner visualization and makes the diagonal visible
fig.update_traces(diagonal_visible=True, showupperhalf=False) 

# 5. SHOW THE PLOT
fig.show()


# ### 7.3 Visualisation of Normal Data

# In[44]:


## Load subset for Visualisation 
df = df_clean2.loc[df_clean2['Anomaly_Label']==1,top_features].sort_values(by='Anomaly_Score', ascending=True)

# 2. DEFINE THE 4 COLUMNS 
columns_to_plot = current_features

# 3. CREATE THE PAIRPLOT (SCATTER MATRIX)
fig = px.scatter_matrix(
    df,
    # Use the 'dimensions' argument to select the columns
    dimensions=columns_to_plot, 
    color='Anomaly_Label',
    color_discrete_sequence='Viridis',
    title="Pairplot (Scatter Matrix) of Selected Columns",
    height=1080,
    width=1080
)

# 4. OPTIONAL: Customize the plot appearance
# Hides the upper triangle for cleaner visualization and makes the diagonal visible
fig.update_traces(diagonal_visible=True, showupperhalf=False) 

# 5. SHOW THE PLOT
fig.show()


# In[ ]:





# In[ ]:




