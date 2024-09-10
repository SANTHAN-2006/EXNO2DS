# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
### STEP 1 :
#### Code : 
```python
# Import the required libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import sklearn for handling outliers
from sklearn.impute import SimpleImputer

# Load the Titanic dataset from seaborn
titanic = pd.read_csv('titanic_dataset.csv')

# Display the first few rows of the dataset to understand its structure
titanic.head()

```
### Output :
![image](https://github.com/user-attachments/assets/2451684b-a2f5-4571-a3e9-dc49f22d1ec4)
<br>
<br>

### STEP 2 :
#### Code :
```python
# Checking for missing values in the dataset
print(titanic.isnull().sum())

# Handling missing values
# For numerical columns, we can use median/mean
imputer_mode = SimpleImputer(strategy='most_frequent')
titanic['Cabin'] = imputer_mode.fit_transform(titanic[['Cabin']])[:, 0]

# For categorical columns, we use mode (most frequent value)
titanic['Embarked'] = imputer_mode.fit_transform(titanic[['Embarked']])[:, 0]

# Check if the null values have been handled
titanic.isnull().sum()

```
### Output :
![image](https://github.com/user-attachments/assets/b55f26b5-743d-477d-b18e-440b253e9f3e)
<br>
<br>

### STEP 3 :
#### Code :
```python
# Plotting a boxplot to visualize outliers in the "age" and "fare" columns
plt.figure(figsize=(10, 5))
sns.boxplot(x=titanic['Age'])
plt.title('Boxplot of Age')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=titanic['Pclass'])
plt.title('Boxplot of Pclass')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=titanic['Fare'])
plt.title('Boxplot of Fare')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=titanic['SibSp'])
plt.title('Boxplot of SibSp')
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x=titanic['Parch'])
plt.title('Boxplot of Parch')
plt.show()

```
### Output :
![image](https://github.com/user-attachments/assets/447f9a19-a63a-40f3-9462-8c94b7c090c7)
<br>
![image](https://github.com/user-attachments/assets/dbf2a908-44ee-43b5-9eed-933bf00dd994)
<br>
![image](https://github.com/user-attachments/assets/1ab9e4b3-68fb-4a1a-b61c-cd815e166f71)
<br>
![image](https://github.com/user-attachments/assets/7e0fdcf7-e5af-4110-a8b6-a4c2684bd17a)
<br>
![image](https://github.com/user-attachments/assets/b61ede62-6522-4bd6-b992-dde177c23a67)
<br>
<br>

### STEP 4 :
#### Code :
```python
# Function to remove outliers using the IQR method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_out = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_out

# Removing outliers from 'age' and 'fare' columns
titanic = remove_outliers(titanic, 'Age')
titanic = remove_outliers(titanic, 'Fare')
titanic = remove_outliers(titanic, 'Pclass')
titanic = remove_outliers(titanic, 'SibSp')
titanic = remove_outliers(titanic, 'Parch')


# Checking the updated shape of the dataset after outlier removal
titanic.shape

```
### Output :
![image](https://github.com/user-attachments/assets/1a66bbc9-8c9b-4c3d-bb76-898f262ea094)
<br>
<br>
### STEP 5 :
#### Code :
```python
# Countplot for categorical data - 'sex'
plt.figure(figsize=(8, 5))
sns.countplot(x='Sex', data=titanic)
plt.title('Count of Passengers by Gender')
plt.show()

# Countplot for 'embarked'
plt.figure(figsize=(8, 5))
sns.countplot(x='Embarked', data=titanic)
plt.title('Count of Passengers by Embarked')
plt.show()

```
### Output :
![image](https://github.com/user-attachments/assets/c5b491ec-84aa-48b7-baa8-355fbd2337bf)
<br>
<br>
### STEP 6 :
#### Code :
```python
# Displot for 'age'
plt.figure(figsize=(8, 5))
sns.displot(titanic['Age'], kde=True)
plt.title('Distribution of Age')
plt.show()

# Displot for 'fare'
plt.figure(figsize=(8, 5))
sns.displot(titanic['Fare'], kde=True)
plt.title('Distribution of Fare')
plt.show()

```
### Output :
![image](https://github.com/user-attachments/assets/434a8e04-b0b8-4354-893b-caf5f6fce535)
<br>
![image](https://github.com/user-attachments/assets/a44dfd26-bad6-4b46-ae1b-524a4d509fed)
<br>
<br>

### STEP 7 :
#### Code :
```python
# Cross-tabulation between 'sex' and 'survived'
cross_tab = pd.crosstab(titanic['Sex'], titanic['Survived'])
print(cross_tab)

# Visualization of the cross-tabulation
cross_tab.plot(kind='bar', stacked=True)
plt.title('Survival Count by Gender')
plt.show()

cross_tab2 = pd.crosstab(titanic['Embarked'], titanic['Survived'])
print(cross_tab2)

# Visualization of the cross-tabulation
cross_tab2.plot(kind='bar', stacked=True)
plt.title('Survival Count by Embarked')
plt.show()
```

### Output :
![image](https://github.com/user-attachments/assets/320a2021-9e31-4762-8eb0-ce37dc9aa9e6)
<br>
![image](https://github.com/user-attachments/assets/b41ea9a3-30d3-46b6-8f90-f0839619099e)

### STEP 8 :
#### Code :
```python
# Heatmap to visualize the correlation between numerical features
plt.figure(figsize=(10, 6))
sns.heatmap(titanic.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

```
### Output :
![image](https://github.com/user-attachments/assets/c9932ba1-d722-4eb1-807a-e29c8e983741)


# RESULT
Therefore, successfully performed Exploratory Data Analysis on the given data set.
