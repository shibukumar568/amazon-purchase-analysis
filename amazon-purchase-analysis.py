import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:\\Users\\shibu\\Downloads\\amazon-purchases-sample2.csv")

# Clean & Add Derived Columns
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Order Month'] = df['Order Date'].dt.to_period('M')
df['Purchase Amount'] = df['Purchase Price Per Unit'] * df['Quantity']

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Show dataframe info (structure, column dtypes, memory usage)
print("\nDataFrame Info:")
print(df.info())

# Display summary stats of numerical columns
print("\nSummary stats:")
print(df.describe())

# Final confirmation message
print("\nCleaned data processed successfully!")

from io import StringIO
import sys

# Capture .info() output properly
buffer = StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()
print(info_str)


# Objective 1: Distribution of Purchase Quantity
plt.figure(figsize=(10, 6))
sns.histplot(df['Quantity'], bins=20, kde=True, color='lightblue', edgecolor='black')
plt.title('Distribution of Purchase Quantity')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Objective 2: Total Purchase Amount by Product Category (Assuming 'Category' exists in the dataset)
plt.figure(figsize=(10, 6))
category_spend = df.groupby('Category')['Purchase Amount'].sum().sort_values(ascending=False).head(10)
category_spend.plot(kind='bar', color='coral', edgecolor='black')
plt.title('Top 10 Product Categories by Total Purchase Amount')
plt.xlabel('Category')
plt.ylabel('Total Purchase Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 3: Average Purchase Amount by State
plt.figure(figsize=(10, 6))
state_avg_spend = df.groupby('Shipping Address State')['Purchase Amount'].mean().sort_values(ascending=False).head(10)
state_avg_spend.plot(kind='bar', color='lightgreen', edgecolor='black')
plt.title('Top 10 States by Average Purchase Amount')
plt.xlabel('State')
plt.ylabel('Average Purchase Amount')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Objective 4: Number of Purchases Over Time (Monthly Trend)
monthly_purchases = df.groupby('Order Month').size()
plt.figure(figsize=(12, 6))
monthly_purchases.plot(kind='line', marker='o', color='teal')
plt.title('Monthly Purchase Trend')
plt.xlabel('Order Month')
plt.ylabel('Number of Purchases')
plt.grid(True)
plt.tight_layout()
plt.show()

# Objective 5: Correlation Heatmap Between Numerical Features
numerical_features = df[['Purchase Amount', 'Purchase Price Per Unit', 'Quantity']]
plt.figure(figsize=(8, 6))
sns.heatmap(numerical_features.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numerical Features')
plt.tight_layout()
plt.show()




