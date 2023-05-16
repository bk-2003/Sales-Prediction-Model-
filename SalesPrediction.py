import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score as R2
from sklearn.model_selection import cross_val_score as CVS
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')
traindata_df = pd.read_csv(r'C:\Users\RK COMPUTERS\Desktop\traindata.csv')
print("printing head of train data ")
print(traindata_df.head().to_string())
testdata_df = pd.read_csv(r'C:\Users\RK COMPUTERS\Desktop\testdata.csv')
print("printing head of test data ")
print(testdata_df.head().to_string())
print(f"Training Dataset (row, col): {traindata_df.shape}\n\nTesting Dataset (row, col): {testdata_df.shape}")
print(traindata_df.info(null_counts=True))
print("describing train data ")
print(traindata_df.describe().to_string())
print("describing test data ")
print(testdata_df.describe().to_string())

# Missing values in ascending order
print("Train Data missing values in ascending order:\n")
print(traindata_df.isnull().sum().sort_values(ascending=True), "\n\n")
print("Test Data missing values in ascending order:\n")
print(testdata_df.isnull().sum().sort_values(ascending=True), "\n\n")

# Value count of both categories
print("Branch Count: \n")
print(traindata_df.Branch.value_counts(), "\n\n")
print("ProductLine Count: \n")
print(traindata_df.ProductLine.value_counts(), "\n\n")

# Mode values for both categories in each dataset
print("Branch: \nMode of test values, Mode of train values:\n",
      [testdata_df['Branch'].mode().values[0], traindata_df['Branch'].mode().values[0]])
print("\nProductLine: \nMode of test values, Mode of train values:\n",
      [testdata_df['ProductLine'].mode().values[0], traindata_df['ProductLine'].mode().values[0]])

# Information regarding both datasets
print("Train data information: \n")
print(traindata_df.info())
print("\n\nTest data information: \n")
print(testdata_df.info())

# list of all the numeric columns
num = traindata_df.select_dtypes('number').columns.to_list()
# list of all the categoric columns
cat = traindata_df.select_dtypes('object').columns.to_list()
# numeric df
BM_num = traindata_df[num]
# categoric df
BM_cat = traindata_df[cat]
print("Print the values in categoric and numeric colums of traindata")
print([traindata_df[cat].value_counts() for category in cat[1:]])

# list of all the numeric columns
num2 = testdata_df.select_dtypes('number').columns.to_list()
# list of all the categoric columns
cat2 = testdata_df.select_dtypes('object').columns.to_list()

# numeric df
BM_num2 = testdata_df[num]
# categoric df
BM_cat2 = testdata_df[cat]
print("Print the values in categoric and numeric colums of traindata")
print([testdata_df[category].value_counts() for category in cat[1:]])

# train data replacement to make data more uniform
traindata_df['ProductLine'].replace(['H and B', 'Health and beauty'], ['Health And Beauty', 'Health And Beauty'],
                                    inplace=True)
print("Printing product line counts after replacement in train data ")
print(traindata_df.ProductLine.value_counts())

# test data replacement to make data more uniform
testdata_df['ProductLine'].replace(['H and B', 'Health and beauty'], ['Health And Beauty', 'Health And Beauty'],
                                   inplace=True)
print("Printing product line counts after replacement in test data ")
print(testdata_df.ProductLine.value_counts())

# Total number of transactions for each Product Line
plt.figure(figsize=(15, 4))
sns.countplot(x=traindata_df.ProductLine, data=traindata_df, palette='mako')
plt.ylabel("No. of Transactions")
plt.title("Train Data")
plt.show()
plt.figure(figsize=(15, 4))
sns.countplot(x=testdata_df.ProductLine, data=testdata_df, palette='hot')
plt.ylabel("No. of Transactions")
plt.title("Test Data")
plt.show()

# Percentage of Male/Female making the transactions
plt.figure(figsize=(25, 7))
plt.pie(traindata_df.Gender.value_counts(), labels=["Male", "Female"], shadow=True, colors=['burlywood', 'navajowhite'],
        startangle=15, explode=[0.2, 0], autopct='%1.2f%%')
plt.legend(traindata_df.Gender, labels=["Male", "Female"], shadow=True, title="Gender:", loc=1)
plt.title("Percentage of Male/Female Transactions in Train Data", loc="right")
plt.show()
plt.figure(figsize=(25, 7))
plt.pie(testdata_df.Gender.value_counts(), labels=["Male", "Female"], shadow=True, colors=['lightcoral', 'firebrick'],
        startangle=15, explode=[0.2, 0], autopct='%1.2f%%')
plt.legend(testdata_df.Gender, labels=["Male", "Female"], shadow=True, title="Gender:", loc=1)
plt.title("Percentage of Male/Female Transactions in Test Data", loc="right")
plt.show()

# Percentage of members and normal Customer Type
plt.figure(figsize=(25, 7))
plt.pie(traindata_df.CustomerType.value_counts(), labels=["Members", "Normal"], shadow=True,
        colors=['crimson', 'palevioletred'], autopct='%1.2f%%')
plt.legend(traindata_df.Gender, labels=["Members", "Normal"], shadow=True, title="Type", loc=1)
plt.title("Percentage of Members and Normal Customers in Train Data")
plt.show()
plt.figure(figsize=(25, 7))
plt.pie(testdata_df.CustomerType.value_counts(), labels=["Members", "Normal"], shadow=True,
        colors=['mediumorchid', 'plum'], autopct='%1.2f%%')
plt.legend(testdata_df.Gender, labels=["Members", "Normal"], shadow=True, title="Type", loc=1)
plt.title("Percentage of Members and Normal Customers in Test Data")
plt.show()

# Mean Gross income for each Product Line
plt.figure(figsize=(12, 6))
sns.barplot(x=traindata_df['ProductLine'], y=traindata_df['GrossIncome'])
plt.xlabel("Product Lines")
plt.ylabel("Mean Gross Income")
plt.title("Gross Income for each Product Line in Train Data")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=testdata_df['ProductLine'], y=testdata_df['GrossIncome'], palette='Spectral')
plt.title("Gross Income for each Product Line in Test Data")
plt.xlabel("Product Lines")
plt.ylabel("Mean Gross Income")
plt.show()

# Mean Gross Income for every Branch
plt.figure(figsize=(10, 5))
sns.barplot(x=traindata_df['Branch'], y=traindata_df['GrossIncome'], palette='YlGnBu')
plt.xlabel("Branch")
plt.ylabel("Mean Gross Income")
plt.title("Gross Income for each Branch in Train Data")
plt.show()

plt.figure(figsize=(10, 5))
sns.barplot(x=testdata_df['Branch'], y=testdata_df['GrossIncome'], palette='BuGn')
plt.title("Gross Income for each Branch in Test Data")
plt.xlabel("Branch")
plt.ylabel("Mean Gross Income")
plt.show()

# Percentage Composition of Payment Methods in the transactions
plt.figure(figsize=(25, 7))
plt.pie(traindata_df.Payment.value_counts(), labels=["EWallet", "Cash", "Credit Card"], shadow=True,
        colors=["m", 'b', 'indigo'], startangle=90, autopct='%1.2f%%')
plt.legend(traindata_df.Payment.value_counts(), labels=["EWallet", "Cash", "Credit Card"], shadow=True,
           title="Payment Method", loc=1)
plt.title("Percentage of Payment Methods in Train Data", loc="right")
plt.show()
plt.figure(figsize=(25, 7))
plt.pie(testdata_df.Payment.value_counts(), labels=["EWallet", "Cash", "Credit Card"], shadow=True,
        colors=['springgreen', 'turquoise', 'slategray'], startangle=90, autopct='%1.2f%%')
plt.legend(testdata_df.Payment.value_counts(), labels=["EWallet", "Cash", "Credit Card"], shadow=True,
           title="Payment Method", loc=1)
plt.title("Percentage of Payment Methods in Test Data", loc="right")
plt.show()

#  Composition Ratio of Gender for every Product Line in Train Data
gender_dummies = pd.get_dummies(traindata_df['Gender'])
gender_dummies.head()
df = pd.concat([traindata_df, gender_dummies], axis=1)

plt.figure(figsize=(12, 6))
sns.barplot(x='ProductLine', y='Female', data=df, palette='winter')
plt.title("Product line of female in train data")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='ProductLine', y='Male', data=df, palette='cool')
plt.title("Product line of male in train data")
plt.show()

#  Composition Ratio of Gender for every Product Line in Test Data
gender_dummies = pd.get_dummies(testdata_df['Gender'])
gender_dummies.head()
df = pd.concat([testdata_df, gender_dummies], axis=1)

plt.figure(figsize=(12, 6))
sns.barplot(x='ProductLine', y='Female', data=df, palette='summer')
plt.title("Product line of female in test data")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='ProductLine', y='Male', data=df, palette='autumn')
plt.title("Product line of male in test data")
plt.show()

#  Composition Ratio of each Branch for every Product Line in Train Data
branch_dummies = pd.get_dummies(traindata_df['Branch'])
branch_dummies.head()
df = pd.concat([traindata_df, branch_dummies], axis=1)
plt.figure(figsize=(12, 6))
sns.barplot(x='ProductLine', y='A', data=df, palette='nipy_spectral')
plt.title("Composition of Branch A(Train Data)")
plt.xlabel("Product Lines")
plt.ylabel("Ratios")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='ProductLine', y='B', data=df, palette='gist_rainbow')
plt.title("Composition of Branch B(Train Data)")
plt.xlabel("Product Lines")
plt.ylabel("Ratios")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='ProductLine', y='C', data=df, palette='YlOrRd')
plt.title("Composition of Branch C(Train Data)")
plt.xlabel("Product Lines")
plt.ylabel("Ratios")
plt.show()

#  Composition Ratio of each Branch for every Product Line in Test Data
branch_dummies = pd.get_dummies(testdata_df['Branch'])
branch_dummies.head()
df = pd.concat([testdata_df, branch_dummies], axis=1)
plt.figure(figsize=(12, 6))
sns.barplot(x='ProductLine', y='A', data=df, palette='nipy_spectral')
plt.title("Composition of Branch A (Test Data)")
plt.xlabel("Product Lines")
plt.ylabel("Ratios")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='ProductLine', y='B', data=df, palette='gist_rainbow')
plt.title("Composition of Branch B(Test Data)")
plt.xlabel("Product Lines")
plt.ylabel("Ratios")
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='ProductLine', y='C', data=df, palette='YlOrRd')
plt.title("Composition of Branch C(Test Data)")
plt.xlabel("Product Lines")
plt.ylabel("Ratios")
plt.show()

# MachineLearning
# load data
data = pd.read_csv(r'C:\Users\RK COMPUTERS\Desktop\completedata.csv')
print(data.head().to_string())

# label encoding
le = LabelEncoder()
Label = ['CustomerType', 'Gender', 'Payment']
for i in Label:
    data[i] = le.fit_transform(data[i])
print(data.head().to_string())

# one hot encoding
cols = ['Branch', 'City', 'ProductLine']
# Apply one-hot encoder
OH_encoder = OneHotEncoder(sparse=False)
data_oh = pd.DataFrame(OH_encoder.fit_transform(data[cols])).astype('int64')

# get feature columns
data_oh.columns = OH_encoder.get_feature_names(cols)

# One-hot encoding removed index; put it back
data_oh.index = data.index

# Add one-hot encoded columns to our main df new name: tr_fe, te_fe (means feature engineered )
data_fe = pd.concat([data, data_oh], axis=1)

# Dropping irrelevant columns

data_fe = data_fe.drop(
    ['Invoice ID', 'Branch', 'City', 'ProductLine', 'Tax 5%', 'Date', 'Time', 'Rating', 'Total', 'cogs',
     'gross margin percentage'], axis=1)
print(data_fe.head())

# LinearRegression
y = data_fe['Quantity']
X = data_fe.drop('Quantity', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)


def cross_val(model_name, model, X, y, cv):
    scores = CVS(model, X, y, cv=cv)
    print(f'{model_name} Scores:')
    for a in scores:
        print(round(a * 100, 2), "%")
    print(f'Average {model_name} score: {round(scores.mean() * 100, 4)} %')


# model
LR = LinearRegression()

# fit
LR.fit(X_train, y_train)

# predict
y_predict = LR.predict(X_test)

# score variables
LR_MAE = round(MAE(y_test, y_predict), 2)
LR_MSE = round(MSE(y_test, y_predict), 2)
LR_R_2 = round(R2(y_test, y_predict), 4)
LR_CS = round(CVS(LR, X, y, cv=5).mean(), 4)

print(f" Mean Absolute Error: {LR_MAE}\n")
print(f" Mean Squared Error: {LR_MSE}\n")
print(f" R^2 Score: {LR_R_2 * 100}%\n")
cross_val(LR, LinearRegression(), X, y, 5)

MAE = [LR_MAE]
MSE = [LR_MSE]
R_2 = [LR_R_2]
Cross_score = [LR_CS]

Models = pd.DataFrame({
    'models': ["Linear Regression"],
    'MAE': MAE, 'MSE': MSE, 'R^2': R_2, 'Cross Validation Score': Cross_score})
Models.sort_values(by='MAE', ascending=True)

# Kmeans Clustering
# standardizing the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_fe)

pd.DataFrame(data_scaled)

# statistics of scaled data
print(pd.DataFrame(data_scaled).describe().to_string())

# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=3, init='k-means++')

# fitting the k means algorithm on scaled data
print(kmeans.fit(data_scaled))

# how well clustering has been done through k-means
print(kmeans.inertia_)

# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1, 20):
    kmeans = KMeans(n_clusters=cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster': range(1, 20), 'SSE': SSE})
plt.figure(figsize=(12, 6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# k means using 5 clusters and k-means++ initialization
kmeans = KMeans(n_clusters=5, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)

frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
print(frame['cluster'].value_counts())
plt.figure(dpi=100)
sns.heatmap(np.round(data.corr(), 2), annot=True)
plt.show()

sns.catplot(y='Rating', x='Quantity', data=data, kind='boxen', aspect=3)
plt.xlabel('Quantity')
plt.ylabel('Rating')
plt.show()

# Finding Which Branch has better sale for a particular product type
plt.figure(dpi=100)
sns.countplot(y='ProductLine', hue="Branch", data=data)
plt.xlabel('Count')
plt.ylabel('Product Type')
plt.show()

