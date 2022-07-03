import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
warnings.filterwarnings('ignore')
plt.style.use('seaborn')

knn=KNeighborsClassifier(n_neighbors=4)
nn = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,100,100), activation='relu', random_state=1 ,max_iter=100)
rf = RandomForestClassifier(n_estimators=70, min_samples_split = 30,bootstrap = True, max_depth = 50, min_samples_leaf = 25)


crimes_2001_2004=pd.read_csv("C:/Users/Riya/Desktop/chi/Chicago_Crimes_2001_to_2004.csv", error_bad_lines=False)
crimes_2005_2007=pd.read_csv("C:/Users/Riya/Desktop/chi/Chicago_Crimes_2005_to_2007.csv",error_bad_lines=False)
crimes_2008_2011=pd.read_csv("C:/Users/Riya/Desktop/chi/Chicago_Crimes_2008_to_2011.csv",error_bad_lines=False)
crimes_2012_2017=pd.read_csv("C:/Users/Riya/Desktop/chi/Chicago_Crimes_2012_to_2017.csv",error_bad_lines=False)

crimes_2001_2004 = crimes_2001_2004.sample(n=50000)
crimes_2005_2007 = crimes_2005_2007.sample(n=50000)
crimes_2008_2011 = crimes_2008_2011.sample(n=50000)
crimes_2012_2017 = crimes_2012_2017.sample(n=50000)
# print(crimes_2001_2004.shape)
data_frames=[crimes_2001_2004, crimes_2005_2007, crimes_2008_2011, crimes_2012_2017]
df=pd.concat(data_frames)

# crimes.info()

# print('Dataset Shape before drop_duplicate : ', crimes.shape)
df.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)
# print('Dataset Shape after drop_duplicate: ', crimes.shape)

df.Date = pd.to_datetime(df.Date, format='%m/%d/%Y %I:%M:%S %p')
# setting the index to be the date will help us a lot later on
df.index = pd.DatetimeIndex(df.Date)

# crimes.info()

loc_to_change  = list(df['Location Description'].value_counts()[20:].index)
# print(loc_to_change)
df.loc[df['Location Description'].isin(loc_to_change) , df.columns=='Location Description'] = 'OTHER'
df['Location Description']
desc_to_change = list(df['Description'].value_counts()[20:].index)
# print(desc_to_change)
df.loc[df['Description'].isin(desc_to_change) , df.columns=='Description'] = 'OTHER'
df['Description']

df = df.dropna()
# crimes.isnull().sum()
df[['District', 'Ward','Community Area']] = df[['District', 'Ward','Community Area']].astype('int')
df[['District', 'Ward','Community Area']] = df[['District', 'Ward','Community Area']].astype('str')

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

df['date2'] = pd.to_datetime(df['date'])
df['Year'] = df['date2'].dt.year
df['Month'] = df['date2'].dt.month
df['Day'] = df['date2'].dt.day
df['Hour'] = df['date2'].dt.hour
df['Minute'] = df['date2'].dt.minute
df['Second'] = df['date2'].dt.second
df = df.drop(['date'], axis=1)
df = df.drop(['date2'], axis=1)

all_classes = df.groupby(['primary_type'])['block'].size().reset_index()

all_classes['amt'] = all_classes['block']
all_classes = all_classes.drop(['block'], axis=1)
all_classes = all_classes.sort_values(['amt'], ascending=[False])
unwanted_classes = all_classes.tail(12)

df.loc[df['primary_type'].isin(unwanted_classes['primary_type']), 'primary_type'] = 'OTHERS'

a=df['primary_type'].unique()

df['block'] = pd.factorize(df["block"])[0]
df['primary_type'] = pd.factorize(df["primary_type"])[0]
df['description'] = pd.factorize(df["description"])[0]
df['location_description'] = pd.factorize(df["location_description"])[0]
df['district'] = pd.factorize(df["district"])[0]
df['ward'] = pd.factorize(df["ward"])[0]
df['community_area'] = pd.factorize(df["community_area"])[0]

x = df.drop(['primary_type'], axis=1)
y = df['primary_type']

features=["description", "arrest"]

from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df ,test_size=0.2, random_state=1)
target='primary_type'
x_train=train_data[features]
y_train=train_data[target]
x_test=test_data[features]
y_test=test_data[target]

#KNN Model
knn.fit(x_train,y_train)

y_predk=knn.predict(x_test)
print("KNN Accuracy    : ", accuracy_score(y_test, y_predk))

#Neural network model
nn.fit(x_train,y_train)
y_predn = nn.predict(x_test)

# accuracy_score(y_test, predicted_result)

print("Neural networks Accuracy    : ", accuracy_score(y_test, y_predn))

#Random forest model
rf.fit(x_train, y_train)
y_predr =rf.predict(x_test)

print("Random forest Accuracy    : ",accuracy_score(y_test, y_predr))


'''
OUTPUT 
KNN Accuracy    :  0.7471286906829514
Neural networks Accuracy    :  0.7675219900274525
Random forest Accuracy    :  0.7668496834556557

Process finished with exit code 0
'''