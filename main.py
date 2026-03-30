import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''Phase 1: Setting up the Data'''
# Load the datasets into pandas
train_df = pd.read_csv('train.csv', low_memory=False)
store_df = pd.read_csv('store.csv')

# Create a local SQLite database connection in your project folder
conn = sqlite3.connect('rossmann_sales.db')

# Push the dataframes into SQL tables
train_df.to_sql('sales', conn, if_exists='replace', index=False)
store_df.to_sql('stores', conn, if_exists='replace', index=False)

# Check shapes
print("Train shape:", train_df.shape)
print("Store shape:", store_df.shape)

# Preview
print(train_df.head())
print(store_df.head())

# Check for nulls
print("\nTrain nulls:\n", train_df.isnull().sum())
print("\nStore nulls:\n", store_df.isnull().sum())

'''Phase 2: SQL Data Cleaning '''
#add helper function
def run_query(sql):
    return pd.read_sql_query(sql, conn)

#First query: top 10 stores by total sales volume
top_stores = run_query("""
    SELECT Store, SUM(Sales) AS TotalSales
    FROM sales
    WHERE Open = 1
    GROUP BY Store
    ORDER BY TotalSales DESC
    LIMIT 10
""")
print(top_stores)

#Second Query: average sales by store type
by_type = run_query("""
    SELECT st.StoreType, 
           ROUND(AVG(s.Sales), 2) AS AvgSales,
           COUNT(*) AS NumRecords
    FROM sales s
    JOIN stores st ON s.Store = st.Store
    WHERE s.Open = 1
    GROUP BY st.StoreType
    ORDER BY AvgSales DESC
""")
print(by_type)

#Third Query: create a view joining sales with competitor distance
conn.execute("""
    CREATE VIEW IF NOT EXISTS sales_with_context AS
    SELECT 
        s.Store,
        s.Date,
        s.Sales,
        s.Customers,
        s.Promo,
        COALESCE(st.StoreType, 'Unknown') AS StoreType,
        st.Assortment,
        st.CompetitionDistance,
        st.Promo2
    FROM sales s
    LEFT JOIN stores st ON s.Store = st.Store
    WHERE s.Open = 1
""")
print("View created!")

# Now query from the view
preview = run_query("SELECT * FROM sales_with_context LIMIT 5")
print(preview)

#Quick data cleaning where train.csv's date column is converted from string to time format
train_df['Date'] = pd.to_datetime(train_df['Date'])

# Also filter out closed store days (Sales = 0 when Open = 0, those skew our stats)
train_open = train_df[train_df['Open'] == 1].copy() # Note: EDA uses df_eda pulled from the SQL view; train_open kept for reference

print("Rows after filtering closed days:", len(train_open))

'''Phase 3: Exploratory Data Analysis'''
#Step 1: Load and clean
# Pull from your view
df_eda = run_query("SELECT * FROM sales_with_context")

# Fix date type (the view returns it as a string)
df_eda['Date'] = pd.to_datetime(df_eda['Date'])

# Fill the ~3 missing CompetitionDistance values with median
median_dist = df_eda['CompetitionDistance'].median()
df_eda['CompetitionDistance'] = df_eda['CompetitionDistance'].fillna(median_dist)
print(f"Filled missing CompetitionDistance with median: {median_dist:.0f}m")

# Verify no nulls remain in key columns
print(df_eda[['Sales', 'CompetitionDistance', 'StoreType']].isnull().sum())

#Step 2: Time Series Feature Extraction
df_eda['Year']       = df_eda['Date'].dt.year
df_eda['Month']      = df_eda['Date'].dt.month
df_eda['DayOfWeek']  = df_eda['Date'].dt.day_name()  # 'Monday', 'Tuesday', etc.
df_eda['WeekOfYear'] = df_eda['Date'].dt.isocalendar().week.astype(int)
df_eda['DayOfMonth'] = df_eda['Date'].dt.day #retail usually sees huge spikes around common paydays (1st and 15th day of month)

print(df_eda[['Date', 'Year', 'Month', 'DayOfWeek', 'Sales']].head())

#Step 3: Core Visualizations
#Set a consistent style so all charts look professional
sns.set_theme(style="whitegrid", palette="muted")

#Chart 1: Promo impact boxplot
fig, ax = plt.subplots(figsize=(8, 5))

sns.boxplot(
    data=df_eda,
    x='Promo',
    y='Sales',
    ax=ax,
    showfliers=False  # Hides extreme outliers so the boxes are readable
)

ax.set_title('Sales Distribution: Promo vs. Non-Promo Days', fontsize=14)
ax.set_xlabel('Promo Active (0 = No, 1 = Yes)')
ax.set_ylabel('Daily Sales (€)')
ax.set_xticks([0, 1])
ax.set_xticklabels(['No Promo', 'Promo'])
plt.tight_layout()
plt.savefig('chart1_promo_impact.png', dpi=150)
plt.show()

#Chart 2: Monthly Seasonality Line Chart
monthly_avg = (
    df_eda.groupby('Month')['Sales']
    .mean()
    .reset_index()
)

fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=monthly_avg, x='Month', y='Sales', marker='o', ax=ax)

ax.set_title('Average Daily Sales by Month (Seasonality)', fontsize=14)
ax.set_xlabel('Month')
ax.set_ylabel('Average Sales (€)')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                    'Jul','Aug','Sep','Oct','Nov','Dec'])
plt.tight_layout()
plt.savefig('chart2_seasonality.png', dpi=150)
plt.show()

#Chart 3: Store type performance bar chart
store_type_avg = (
    df_eda.groupby('StoreType')['Sales']
    .mean()
    .reset_index()
    .sort_values('Sales', ascending=False)
)

fig, ax = plt.subplots(figsize=(7, 5))
sns.barplot(data=store_type_avg, x='StoreType', y='Sales', ax=ax)

ax.set_title('Average Daily Sales by Store Type', fontsize=14)
ax.set_xlabel('Store Type')
ax.set_ylabel('Average Sales (€)')
plt.tight_layout()
plt.savefig('chart3_store_type.png', dpi=150)
plt.show()

#Promo uplift
avg_promo    = df_eda[df_eda['Promo'] == 1]['Sales'].mean()
avg_no_promo = df_eda[df_eda['Promo'] == 0]['Sales'].mean()
uplift = (avg_promo - avg_no_promo) / avg_no_promo * 100

print(f"Promo Uplift: {uplift:.1f}%")
# Calculate Uplift by Store Type
promo_by_type = df_eda.groupby(['StoreType', 'Promo'])['Sales'].mean().unstack()
# unstack() creates columns: 0 (No Promo) and 1 (Promo)

promo_by_type['Uplift_Percent'] = ((promo_by_type[1] - promo_by_type[0]) / promo_by_type[0]) * 100

print("\nPromo Uplift by Store Type:")
print(promo_by_type[['Uplift_Percent']].sort_values('Uplift_Percent', ascending=False))

#Coefficient of Variation (CV) per store
cv_by_store = (
    df_eda.groupby('Store')['Sales']
    .agg(Mean='mean', Std='std')
    .assign(CV=lambda x: x['Std'] / x['Mean'])
    .sort_values('CV', ascending=False)
)

print("Most volatile stores (High CV = hard to manage inventory):")
print(cv_by_store.head(10))

print("\nMost stable stores (Low CV = easy to manage inventory):")
print(cv_by_store.tail(10))

# Chart 4: Supply Chain Volatility (CV vs Volume)
fig, ax = plt.subplots(figsize=(8, 6))

# Plotting Mean Sales (Volume) vs CV (Volatility)
sns.scatterplot(
    data=cv_by_store,
    x='Mean',
    y='CV',
    alpha=0.6, # Transparency helps see overlapping dots
    color='purple',
    ax=ax
)

# Draw a line showing the "average" CV for context
median_cv = cv_by_store['CV'].median()
ax.axhline(median_cv, color='red', linestyle='--', label=f'Median CV ({median_cv:.2f})')

ax.set_title('Store Profiling: Sales Volume vs. Demand Volatility', fontsize=14)
ax.set_xlabel('Average Daily Sales (Volume)')
ax.set_ylabel('Coefficient of Variation (Volatility)')
ax.legend()
plt.tight_layout()
plt.savefig('chart4_volatility_scatter.png', dpi=150)
plt.show()

df_eda.to_csv('rossmann_final_cleaned.csv', index=False) #ensures that the EDA travels into Power BI
conn.close() #closes database