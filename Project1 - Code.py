#Assignment - Data Preparation

#Code:
import statistics
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew, kurtosis
file = "Assignment.csv"
data = pd.read_csv(file)


# Q1: What is the median income of the dataset? Enter values round up to two decimal points
#Code:
median_income = data['Income'].median().round(2) 
print(f"Median income of the dataset: {median_income}")
#Solution: Median income of the dataset: 65331.50



# Q2: How many unique job types are present in the dataset? Enter values round up to two decimal points.
#Code:
unique_job_types_count = data['Job Type'].nunique()
print(f'Number of unique job types: {str(round(unique_job_types_count, 2))}')
#Solution: Number of unique job types: 4.00



# Q3: Calculate the average number of dependents per customer. Enter values round up to two decimal points.
#Code:
average_dependents = data['Number of Dependents'].mean().round(2)
print(f"Average number of dependents per customer: {average_dependents}")
#Solution: Average number of dependents per customer: 2.20



#Q4: What is the maximum credit score recorded in the dataset? Enter values round up to two decimal points.
#Code:
max_credit_score = data['Credit Score'].max().round(2)
print(f"Maximum credit score recorded in the dataset: {max_credit_score}")
#Solution: Maximum credit score recorded in the dataset: 758.55



#Q5: Count the number of customers who have a car, enter values round up to two decimal points.
#Code:
customers_with_car_count = data[data['Has Car'] == 'Yes'].shape[0] 
print(f"Number of customers who have a car: {customers_with_car_count:.2f}")
#Solution: Number of customers who have a car: 104.00


#Q6: What is the total number of missing values in the 'Years in Current Job' column? Enter values round up to two decimal points.
#Code:
missing_values_count = data['Years in Current Job'].isna().sum().round(2) 
print(f"Total number of missing values in the 'Years in Current Job' column: {missing_values_count}")
#Solution: Total number of missing values in the 'Years in Current Job' column: 20.00



#Q7: Find the standard deviation of the 'Annual Spending' for the dataset. Enter values round up to two decimal points.
#Code:
std_dev_annual_spending = data['Annual Spending'].std().round(2) 
print(f"Standard deviation of the 'Annual Spending': {std_dev_annual_spending}")
#(or)
Stdev = np.std(data ['Annual Spending'], ddof=1).round(2)
print(Stdev)
#Solution: Standard deviation of the 'Annual Spending': 9833.23



#Q8: How many customers are classified as 'High' in customer satisfaction?
#Code:
high_satisfaction_count = data[data['Customer Satisfaction'] == 'High'].shape[0]  
print(f"Number of customers classified as 'High' in customer satisfaction: {high_satisfaction_count}")
#Solution: Number of customers classified as 'High' in customer satisfaction: 63



#Q9: What is the mode of the 'Education Level' in the dataset?
#Code:
education_level_mode = data['Education Level'].mode()[0]
print(f"Mode of the 'Education Level': {education_level_mode}")
#Solution: Mode of the 'Education Level': Master



#Q10: Calculate the average age of customers who do not have a car. Enter integer value
#Code:
customers_without_car = data[data['Has Car'] == 'No']
average_age_without_car = int(customers_without_car['Age'].mean())
print(f"Average age of customers who do not have a car: {average_age_without_car}")
#Solution: Average age of customers who do not have a car: 40



#Q11: Which column has the highest number of outliers?
#Code:
# Define a function to detect outliers using the IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers)
# Apply the function to all numerical columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
outliers_count = {col: detect_outliers_iqr(data, col) for col in numerical_columns}
# Find the column with the highest number of outliers
max_outliers_column = max(outliers_count, key=outliers_count.get)
max_outliers_value = outliers_count[max_outliers_column]
print(f"Column with the highest number of outliers: {max_outliers_column}")
print(f"Number of outliers in {max_outliers_column}: {max_outliers_value}")
#Solution: Column with the highest number of outliers: Income; Number of outliers in Income: 7 



#Q12: What is the most common type of job among the customers?
#Code:
most_common_job = data['Job Type'].mode()[0]
print(f"Most Common Job type: {most_common_job}")
#Solution: Most Common Job type: Part-time. Also, as per Excel analysis Most Common Job Type using Mode.Sngl function: Unemployed.



#Q13: Which variable is most likely to have a normal distribution?
#Method: To determine which variable is most likely to have a normal distribution, we need to analyze the distribution of numerical variables in the dataset. 
# A common approach is to visualize the distributions using histograms or density plots and to calculate statistical measures such as skewness and kurtosis. 
# Variables with skewness close to 0 and kurtosis close to 3 are more likely to be normally distributed.
#Code: 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
# Select numerical columns
numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
# Analyze skewness and kurtosis for each numerical column
for col in numerical_cols:
    print(f"\nColumn: {col}")
    print(f"Skewness: {skew(data[col].dropna()):.2f}")
    print(f"Kurtosis: {kurtosis(data[col].dropna()):.2f}")
    
    # Plot histogram and density plot
    plt.figure(figsize=(8, 4))
    sns.histplot(data[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()
#Solution: The variable most likely to have a normal distribution is: "Purchase Frequency" as per Skewness value. 
# However, as per visual inspection "Credit Score" seems to have normal distribution too.



#Q14: What percentage of the dataset is missing the 'Income' data?
#Code:    
missing_income_percentage = (data['Income'].isna().sum() / len(data)) * 100
print(f"Percentage of missing 'Income' data: {missing_income_percentage}%")
#Solution: Percentage of missing 'Income' data: 10.0%



#Q15: Which age group shows the highest median income?
# Define age groups
bins = [18, 30, 45, 60, 75] 
labels = ['18-30', '31-45', '46-60', '61-75']
# Create an 'Age Group' column based on the age bins
data['Age Group'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
# Calculate the median income for each age group
median_income_by_age_group = data.groupby('Age Group')['Income'].median()
# Find the age group with the highest median income
highest_median_income_group = median_income_by_age_group.idxmax()
highest_median_income_value = median_income_by_age_group.max()
print(f"Age group with the highest median income: {highest_median_income_group} with a median income of {highest_median_income_value}")
#Solution: Age group with the highest median income: 61-75 with a median income of 84877.0



#Q21: Replace the missing values in income with median, education level with mode, years in current job with mode, what is the standard deviation of income?
#Code:
std_dev_income = data['Income'].std().round(2) 
print(f"Standard deviation of the 'Income' before replacing missing values: {std_dev_income}")
# Enter values round up to two decimal points.
# For 'Income', replace missing values with the median
income_median = data['Income'].median()
# For 'Education Level', replace missing values with the mode
education_mode = data['Education Level'].mode()[0]
# For 'Years in Current Job', replace missing values with the mode
years_in_current_job_mode = data['Years in Current Job'].mode()[0]
# Fill missing values for each column individually
data['Income'] = data['Income'].fillna(income_median)
data['Education Level'] = data['Education Level'].fillna(education_mode)
data['Years in Current Job'] = data['Years in Current Job'].fillna(years_in_current_job_mode)
#standard deviation of income with newly filled values.
std_dev_income = data['Income'].std().round(2) 
print(f"Standard deviation of the 'Income' after replacing missing values: {std_dev_income}")
#Solution: Standard deviation of the 'Income' before replacing missing values: 35956.47; Standard deviation of the 'Income' after replacing missing values: 34128.87



#Q22:
#Code: 
from sklearn.preprocessing import MinMaxScaler
#Initialize the MinMaxScaler
scaler = MinMaxScaler()
# Apply the MinMaxScaler to the 'Income' column
data['Income_scaled'] = scaler.fit_transform(data[['Income']])
# Calculate the standard deviation of the scaled 'Income' column
income_scaled_std_dev = data['Income_scaled'].std()
# Round the result to two decimal points
income_scaled_std_dev_rounded = round(income_scaled_std_dev, 2)
print(f"Standard deviation of scaled 'Income': {income_scaled_std_dev_rounded}")
#Solution: Standard deviation of scaled 'Income': 0.16




