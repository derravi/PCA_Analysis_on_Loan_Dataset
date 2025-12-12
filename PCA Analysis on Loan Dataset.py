#Import all the Requered Library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Letse see the All the details of the Datasets......\n")
df = pd.read_csv("loan.csv")
df.head()

print("Let see there is any Null values are present or not into the datasets.........\n")
df.isnull().sum()

print("Letse Remove all the null values into this datasets........\n")
l1=['Gender','Married','Dependents','Self_Employed']
l2=['LoanAmount','Loan_Amount_Term','Credit_History']

for i in l1:
    df[i] = df[i].fillna(df[i].mode()[0])
for j in l2:
    df[j]= df[j].fillna(df[j].mean())

print("Letse Again check there is any null values are present or not.........\n")
df.isnull().sum()

print("Lets Descrive the all the column of the datsets...........\n")
df.describe(include='all')

print("Lets Rescall the requered Columsn into the datsets.........\n")
scaler = StandardScaler()
x = df[['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]

new_df = scaler.fit_transform(x)
pca = PCA(n_components=5)

ndf = pca.fit_transform(new_df)

nndf = pd.DataFrame(ndf,columns=['PCA1','PCA2','PCA3','PCA4','PCA5'])

variance = pca.explained_variance_ratio_

print("The variance of this data is:")
print(np.round(variance*100,2))

print("Lets see the PCA Analysis for this Datsets.........\n")

plt.figure(figsize=(15,7))


plt.scatter(nndf['PCA1'], nndf['PCA2'],color="red", edgecolor='black', label='PCA1 vs PCA2')
plt.scatter(nndf['PCA2'], nndf['PCA3'],color="green", edgecolor='black', label='PCA2 vs PCA3')
plt.scatter(nndf['PCA3'], nndf['PCA4'],color="blue", edgecolor='black', label='PCA3 vs PCA4')
plt.scatter(nndf['PCA4'], nndf['PCA5'],color="yellow", edgecolor='black', label='PCA4 vs PCA5')
plt.title("PCA Analysis")
plt.xlabel("PCA Component (X-axis)")
plt.ylabel("PCA Component (Y-axis)")
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig("PCA_Analysys.png", dpi=500, bbox_inches='tight')
plt.show()


print("")

l1 = ['PCA1','PCA2','PCA3','PCA4','PCA5']
values=np.round(variance*100,2)
plt.pie(values,labels=l1,autopct="%1.1f%%",startangle=90)
plt.title("PCA Analysys")
plt.show()

