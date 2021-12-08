import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('covtype.csv', index_col=0)

# Class distribution
a = dataset['Cover_Type']
sns.countplot(a,palette=["#AFEEEE","#B0C4DE"])

# Correlation
size = 10
data = dataset.iloc[:, :size]
cols = data.columns
data_corr = data.corr()
threshold = 0.5
corr_list = []
for i in range(0, size):
  for j in range(i+1, size):
    if (data_corr.iloc[i, j] >= threshold and data_corr.iloc[i, j] < 1) or (data_corr.iloc[i, j] < 0 and data_corr.iloc[i, j] <= -threshold):
       corr_list.append([data_corr.iloc[i, j], i, j])
s_corr_list = sorted(corr_list, key=lambda x: -abs(x[0]))

# Scatter Plot (Pair plot)
for v, i, j in s_corr_list:
  sns.pairplot(dataset, hue="Cover_Type", height=6, x_vars=cols[i], y_vars=cols[j])
  sns.color_palette("flare", as_cmap=True)
  plt.show()

# Heat Map
col_list = dataset.columns
col_list = [col for col in col_list if not col[0:4] == 'Soil']
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(dataset[col_list].corr(), square=True, linewidths=1, cmap="Blues")
plt.title('Correlation of Variables')
plt.show()

# Horizontal_Distance_To_Hydrology & Vertical_Distance_To_Hydrology with Soil_Type1
sns.lmplot(x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', data=dataset, hue='Soil_Type1', fit_reg=False, palette="Blues_r")

# Horizontal_Distance_To_Hydrology & Vertical_Distance_To_Hydrology with Wilderness_Area2
sns.lmplot(x='Horizontal_Distance_To_Hydrology', y='Vertical_Distance_To_Hydrology', data=dataset, hue='Wilderness_Area2',fit_reg=False, palette="Blues_r")
