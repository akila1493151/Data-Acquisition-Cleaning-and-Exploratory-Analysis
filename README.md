import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting styles
sns.set_theme(style="whitegrid")

# -------------------------------------------------------------------------
# Setup: Create a representative Raw Dataset if not already loaded
# -------------------------------------------------------------------------
np.random.seed(42)
n_rows = 500

# Generating raw synthetic variables matching the tasks
data = {
    'Timestamp': pd.date_range(start='2026-01-01', periods=n_rows, freq='h').strftime('%Y-%m-%d %H:%M:%S'),
    'Feature_Skewed': np.random.exponential(scale=10, size=n_rows),  # Highly positively skewed
    'Feature_Linear': np.random.normal(loc=50, scale=10, size=n_rows),
    'Feature_Nonlinear': np.zeros(n_rows),  # Will populate below
    'Raw_Numeric_as_Object': np.random.choice(['10.5', '20.3', '15.2', 'unknown', '30.1'], size=n_rows),
    'Repetitive_String': np.random.choice(['Low', 'Medium', 'High'], size=n_rows, p=[0.5, 0.3, 0.2]),
    'Target_Value': np.random.normal(loc=100, scale=25, size=n_rows)
}
df_raw = pd.DataFrame(data)
# Create a strict monotonic non-linear relationship: y = x^3
df_raw['Feature_Nonlinear'] = df_raw['Feature_Linear'] ** 3 + np.random.normal(0, 1000, n_rows)

# Introduce systematic Null values
df_raw.loc[df_raw.sample(frac=0.25, random_state=1).index, 'Raw_Numeric_as_Object'] = np.nan # >20%
df_raw.loc[df_raw.sample(frac=0.08, random_state=2).index, 'Feature_Skewed'] = np.nan        # <20%
df_raw.loc[df_raw.sample(frac=0.05, random_state=3).index, 'Feature_Linear'] = np.nan        # <20%

# Inject duplicate rows intentionally
duplicates = df_raw.iloc[:15].copy()
df = pd.concat([df_raw, duplicates], ignore_index=True)


# =========================================================================
# Task 1: Dataset Ingestion & Structure Audit
# =========================================================================
print("=== Task 1: Dataset Ingestion ===")
print(f"DataFrame Shape: {df.shape}")
print("\n--- First 5 Rows ---")
print(df.head(5))
print("\n--- Column Data Types ---")
print(df.dtypes)
print("\n" + "="*50 + "\n")


# =========================================================================
# Task 2: Null Value Analysis & Conditional Threshold Imputation
# =========================================================================
print("=== Task 2: Null Value Analysis ===")
null_counts = df.isnull().sum()
null_percentages = (null_counts / df.shape[0]) * 100
null_df = pd.DataFrame({'Null Count': null_counts, 'Percentage (%)': null_percentages})
print(null_df)

high_null_cols = null_percentages[null_percentages > 20].index.tolist()
print(f"\nColumns exceeding 20% null rate (handled via modeling pipeline later): {high_null_cols}")

# Strategy: For numeric columns < 20% nulls, fill with median (except the 2 most skewed columns handled in Task 9a)
# We will identify numeric columns here but leave the 2 most skewed for explicit side-by-side handling later.
print("\n" + "="*50 + "\n")


# =========================================================================
# Task 3: Duplicate Detection and Removal
# =========================================================================
print("=== Task 3: Duplicate Detection & Removal ===")
initial_rows = df.shape[0]
dup_count = df.duplicated().sum()
print(f"Number of duplicate rows detected: {dup_count}")

df = df.drop_duplicates().reset_index(drop=True)
removed_rows = initial_rows - df.shape[0]
print(f"Rows removed: {removed_rows} | New DataFrame Shape: {df.shape}")
print("\n" + "="*50 + "\n")


# =========================================================================
# Task 4: Data Type Correction & Memory Optimization
# =========================================================================
print("=== Task 4: Data Type Correction ===")
mem_before = df.memory_usage(deep=True).sum()

# 1. Inferred object to numeric conversion safely casting errors to NaN
df['Raw_Numeric_as_Object'] = pd.to_numeric(df['Raw_Numeric_as_Object'], errors='coerce')
# 2. Repetitive string column to Category
df['Repetitive_String'] = df['Repetitive_String'].astype('category')

mem_after = df.memory_usage(deep=True).sum()
print(f"Memory usage BEFORE optimization: {mem_before:,} bytes")
print(f"Memory usage AFTER optimization:  {mem_after:,} bytes")
print(f"Absolute reduction: {mem_before - mem_after:,} bytes ({(1 - mem_after/mem_before)*100:.2f}% Saved)")
print("\n" + "="*50 + "\n")


# =========================================================================
# Task 5: Descriptive Statistics & Skewness Profiling
# =========================================================================
print("=== Task 5: Descriptive Statistics & Skewness ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(df[numeric_cols].describe())

skew_dict = {col: df[col].skew() for col in numeric_cols}
skew_df = pd.DataFrame.from_dict(skew_dict, orient='index', columns=['Skewness'])
print("\n--- Skewness Metrics ---")
print(skew_df)

highest_skew_col = skew_df['Skewness'].abs().idxmax()
print(f"\nColumn with highest absolute skewness: {highest_skew_col} (Value: {skew_dict[highest_skew_col]:.4f})")
print("\n" + "="*50 + "\n")


# =========================================================================
# Task 6: Outlier Detection via Interquartile Range (IQR)
# =========================================================================
print("=== Task 6: Outlier Detection with IQR ===")
outlier_targets = ['Feature_Linear', 'Target_Value']

for col in outlier_targets:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Column: {col:15} | IQR: {IQR:6.2f} | Bounds: [{lower_bound:.2f}, {upper_bound:.2f}] | Outliers detected: {len(outliers)} rows")
print("\n" + "="*50 + "\n")


# =========================================================================
# Task 7: Data Visualization Suite
# =========================================================================
fig, axes = plt.subplots(3, 2, figsize=(14, 18))
fig.suptitle('Exploratory Data Analysis Data Profile Suite', fontsize=16, weight='bold')

# Plot 1: Line Plot (Chronological or sorting profile)
# Converting Timestamp to datetime for safe sorting
df_sorted = df.copy()
df_sorted['Timestamp'] = pd.to_datetime(df_sorted['Timestamp'])
df_sorted = df_sorted.sort_values(by='Timestamp')
axes[0, 0].plot(df_sorted['Timestamp'], df_sorted['Target_Value'], color='tab:blue', alpha=0.7)
axes[0, 0].set_title('1. Target Value Sequence Profile Line Plot')
axes[0, 0].set_xlabel('Timeline Index')
axes[0, 0].set_ylabel('Target Value')
axes[0, 0].tick_params(axis='x', rotation=30)

# Plot 2: Categorical Bar Chart
grouped_bar = df.groupby('Repetitive_String', observed=False)['Feature_Linear'].mean()
axes[0, 1].bar(grouped_bar.index.astype(str), grouped_bar.values, color=['#4C72B0', '#55A868', '#C44E52'])
axes[0, 1].set_title('2. Mean Feature_Linear across Categories')
axes[0, 1].set_xlabel('Repetitive String Categories')
axes[0, 1].set_ylabel('Mean Feature_Linear Value')

# Plot 3: Histogram of the most skewed numeric column
sns.histplot(data=df, x=highest_skew_col, bins=20, kde=True, ax=axes[1, 0], color='purple')
axes[1, 0].set_title(f'3. Distribution Histogram of Highly Skewed: {highest_skew_col}')
axes[1, 0].set_xlabel(highest_skew_col)
axes[1, 0].set_ylabel('Frequency Count')

# Plot 4: Scatter Plot (Checking correlation)
sns.scatterplot(data=df, x='Feature_Linear', y='Feature_Nonlinear', ax=axes[1, 1], alpha=0.6, color='teal')
axes[1, 1].set_title('4. Scatter Plot: Feature_Linear vs Feature_Nonlinear')
axes[1, 1].set_xlabel('Feature_Linear')
axes[1, 1].set_ylabel('Feature_Nonlinear')

# Plot 5: Categorical Box Plot
sns.boxplot(data=df, x='Repetitive_String', y='Target_Value', ax=axes[2, 0], palette='pastel')
axes[2, 0].set_title('5. Target Value Variance Across Categories Boxplot')
axes[2, 0].set_xlabel('Repetitive String Categories')
axes[2, 0].set_ylabel('Target Value')

# Clean up empty panel axis
fig.delaxes(axes[2, 1])
plt.tight_layout()
plt.show()


# =========================================================================
# Task 8: Pearson Correlation Heat Map Analysis
# =========================================================================
print("=== Task 8: Pearson Correlation Matrix ===")
# Dynamically collect all available real numerical features
numeric_df = df.select_dtypes(include=[np.number])
pearson_corr = numeric_df.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(pearson_corr, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
plt.title("Pearson Linear Product-Moment Correlation Matrix Heatmap")
plt.show()

# Find highest off-diagonal absolute value
corr_matrix_abs = pearson_corr.abs()
np.fill_diagonal(corr_matrix_abs.values, 0)
highest_corr_pair = corr_matrix_abs.unstack().idxmax()
print(f"Highest absolute correlated pair (Pearson): {highest_corr_pair} (Value: {pearson_corr.loc[highest_corr_pair[0], highest_corr_pair[1]]:.4f})\n")


# =========================================================================
# Task 9a: Advanced Imputation Comparison Strategy
# =========================================================================
print("=== Task 9a: Advanced Imputation Strategy Comparison ===")
# Isolate the top two skewed columns
top_2_skewed = skew_df['Skewness'].abs().nlargest(2).index.tolist()

for col in top_2_skewed:
    mean_val = df[col].mean()
    median_val = df[col].median()
    print(f"Column: {col:20} | Pre-Imputation Mean: {mean_val:8.4f} | Pre-Imputation Median: {median_val:8.4f}")
    
    # Impute missing records with safe fallback to column median
    df[col] = df[col].fillna(median_val)

# Verify null counts on these targets drop to zero
print(f"\nRemaining Null Count validation post-imputation:\n{df[top_2_skewed].isnull().sum()}")

# Fill remaining non-skewed numeric columns below 20% null threshold with median
for c in numeric_cols:
    if df[c].isnull().sum() > 0 and (df[c].isnull().sum() / len(df) <= 0.20):
        df[c] = df[c].fillna(df[c].median())

print("\n" + "="*50 + "\n")


# =========================================================================
# Task 9b: Spearman Rank Monotonicity Profiling
# =========================================================================
print("=== Task 9b: Spearman Rank Correlation Matrix ===")
spearman_corr = numeric_df.corr(method='spearman')

# Generate the raw mathematical delta absolute table
diff_matrix = (spearman_corr - pearson_corr).abs()
np.fill_diagonal(diff_matrix.values, 0)

# Extract unique pairs without duplicates or self-comparison
diff_unstacked = diff_matrix.unstack().drop_duplicates().sort_values(ascending=False)
# Remove any entry that represents self-pairing indices (0 absolute distance on the diagonal)
diff_unstacked = diff_unstacked[diff_unstacked > 0.0]

print("\nTop Column Pairs by Absolute Difference (|Spearman - Pearson|):")
diff_table_records = []
for index, delta in diff_unstacked.head(3).items():
    p_val = pearson_corr.loc[index[0], index[1]]
    s_val = spearman_corr.loc[index[0], index[1]]
    diff_table_records.append({
        'Variable A': index[0], 'Variable B': index[1],
        'Pearson': round(p_val, 4), 'Spearman': round(s_val, 4),
        '|Delta|': round(delta, 4)
    })
print(pd.DataFrame(diff_table_records).to_string(index=False))
print("\n" + "="*50 + "\n")


# =========================================================================
# Task 9c: Grouped Variance Aggregation Analysis
# =========================================================================
print("=== Task 9c: Grouped Aggregation Profile ===")
grouped_stats = df.groupby('Repetitive_String', observed=False)['Target_Value'].agg(['mean', 'std', 'count'])
print(grouped_stats)

max_mean_group = grouped_stats['mean'].idxmax()
max_std_group = grouped_stats['std'].idxmax()
mean_ratio = grouped_stats['mean'].max() / grouped_stats['mean'].min()

print(f"\nGroup with highest Mean: {max_mean_group} ({grouped_stats.loc[max_mean_group, 'mean']:.2f})")
print(f"Group with highest StdDev: {max_std_group} ({grouped_stats.loc[max_std_group, 'std']:.2f})")
print(f"Ratio of Highest to Lowest Group Mean: {mean_ratio:.4f}")
print("\n" + "="*50 + "\n")


# =========================================================================
# Task 10: Serialize Output State to File Assets
# =========================================================================
df.to_csv('cleaned_data.csv', index=False)
print("SUCCESS: Cleaned dataset serialized to target 'cleaned_data.csv' ready for model phases.")
