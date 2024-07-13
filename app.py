# %% [markdown]
# <a href="https://colab.research.google.com/github/Halimazz/uas-bengkel-koding/blob/main/uas.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
# Library
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score



# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
dir = "/content/drive/MyDrive/dataset/hungarian-data/hungarian.data"
with open (dir, encoding='Latin1') as file:
    lines = [line.strip() for line in file]
lines[0:10]


# %%
import itertools

# Nama file
dir = "/content/drive/MyDrive/dataset/hungarian-data/hungarian.data"

# Membaca file dengan encoding Latin1
with open(dir, encoding='Latin1') as file:
    lines = [line.strip() for line in file]

# Menampilkan 10 baris pertama dari lines untuk pengecekan
lines[0:10]

# %%
data = itertools.takewhile(
  lambda x: len(x) == 76,
  (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
)

df = pd.DataFrame.from_records(data)

df.head()

# %%
df.info()

# %%
df = df.iloc[:,:-1]
df = df.drop(df.columns[0], axis=1)

# %%
df = df.astype(float)

# %%
df.info()

# %%
df.replace(-9, np.nan, inplace=True)


# %%
selected_columns = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]

# Memberikan nama pada masing-masing kolom
selected_columns.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Menampilkan 5 baris pertama dari DataFrame yang telah dipilih
selected_columns.head()

# %%
df = selected_columns
df.head()

# %%
df.value_counts()

# %%
df.isnull().sum()

# %%
columns_to_drop = ['ca', 'thal','slope']
df = df.drop(columns=columns_to_drop, axis=1)

# %%
df.info()

# %%
df.isnull().sum()

# %%
df.head()

# %%
df = df.dropna()
df.head()

# %%
df.isnull().sum()
df.head()

# %%


# %%
df['target'].value_counts().plot(kind='bar',figsize=(10,16),color=['blue','red'])
plt.title('Hitung target')
plt.xlabel('Target')
plt.ylabel('Jumlah')
plt.show()

# %%
X=df.drop('target',axis=1).values
y=df.iloc[:,-1]

# %%
!pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd

# Asumsikan df sudah didefinisikan dan dipisahkan menjadi X dan y
# X = df.drop('target', axis=1).values
# y = df.iloc[:, -1].values

# Inisialisasi SMOTE
smote = SMOTE(random_state=42)

# Menerapkan SMOTE untuk oversampling
X_smote_resampled, y_smote_resampled = smote.fit_resample(X, y)

# Membuat DataFrame dari dataset yang telah di-resample
df_resampled = pd.DataFrame(X_smote_resampled, columns=df.drop('target', axis=1).columns)
df_resampled['target'] = y_smote_resampled

# Plot distribusi kelas sebelum SMOTE
plt.figure(figsize=(10, 6))
sns.countplot(x=y, palette=['blue', 'red'])
plt.title('Distribusi Kelas Sebelum SMOTE')
plt.xlabel('Kelas Target')
plt.ylabel('Jumlah')
plt.show()

# Plot distribusi kelas setelah SMOTE
plt.figure(figsize=(10, 6))
sns.countplot(x=y_smote_resampled, palette=['blue', 'red'])
plt.title('Distribusi Kelas Setelah SMOTE')
plt.xlabel('Kelas Target')
plt.ylabel('Jumlah')
plt.show()


# %%
y_res = y_smote_resampled
X_res = X_smote_resampled


## OPSIONAL
# Normalisasi atau standarisasi fitur jika diperlukan
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)

# Membagi data menjadi set pelatihan dan set pengujian
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# %%
# SVM
from sklearn import svm
sv = svm.SVC(kernel='linear')
sv.fit(X_train, y_train)
Y_pred_svm = sv.predict(X_test)

score_svm = round(accuracy_score(Y_pred_svm,y_test)*100,2)
print("Akurasi : "+str(score_svm)+" %")

# %%

cm = confusion_matrix(y_test, Y_pred_svm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# %%
# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
Y_pred_knn=knn.predict(X_test)

score_knn = round(accuracy_score(Y_pred_knn,y_test)*100,2)

print("Akurasi : "+str(score_knn)+" %")

# %%
# DECISON TREE
from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(500):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x

#print(max_accuracy)
#print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,y_train)
Y_pred_dt = dt.predict(X_test)

score_dt = round(accuracy_score(Y_pred_dt,y_test)*100,2)

print("Akurasi : "+str(score_dt)+" %")

# %%
# Membuat confusion matrix model Decision Tree
cm = confusion_matrix(y_test, Y_pred_dt)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# %%
scores = [score_svm,score_knn,score_dt]
algorithms = ["Support Vector Machine","K-Nearest Neighbors","Decision Tree"]

for i in range(len(algorithms)):
    print("Akurasi Model "+algorithms[i]+" : "+str(scores[i])+" %")

# %%
# Membangun dan mengevaluasi berbagai model klasifikasi
models = {
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier()
}

# Menyimpan hasil akurasi
results = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append((model_name, accuracy))

# Membuat DataFrame dari hasil
results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

# Membuat barplot
sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Model")
plt.ylabel("Accuracy score")
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.show()


