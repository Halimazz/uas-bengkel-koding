# Library
import itertools
import pandas as pd
import numpy as np
import streamlit as st


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# Streamlit configuration
st.title('Heart Disease Prediction')
st.write("""
    This application uses machine learning models to predict heart disease based on the Hungarian dataset.
""")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["data"])
if uploaded_file is not None:
    lines = [line.decode('Latin1').strip() for line in uploaded_file]
    
    # Process data
    data = itertools.takewhile(
        lambda x: len(x) == 76,
        (' '.join(lines[i:(i + 10)]).split() for i in range(0, len(lines), 10))
    )
    
    df = pd.DataFrame.from_records(data)
    df = df.iloc[:,:-1]
    df = df.drop(df.columns[0], axis=1)
    df = df.astype(float)
    df.replace(-9, np.nan, inplace=True)
    
    selected_columns = df.iloc[:, [1, 2, 7, 8, 10, 14, 17, 30, 36, 38, 39, 42, 49, 56]]
    selected_columns.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = selected_columns.dropna()
    
    st.write("### Data Overview")
    st.write(df.head())
    
    st.write("### Target Distribution")
    fig, ax = plt.subplots()
    df['target'].value_counts().plot(kind='bar', ax=ax, figsize=(10, 6), color=['blue', 'red'])
    ax.set_title('Hitung target')
    ax.set_xlabel('Target')
    ax.set_ylabel('Jumlah')
    st.pyplot(fig)
    
    X = df.drop('target', axis=1).values
    y = df['target']
    
    smote = SMOTE(random_state=42)
    X_smote_resampled, y_smote_resampled = smote.fit_resample(X, y)
    
    df_resampled = pd.DataFrame(X_smote_resampled, columns=df.drop('target', axis=1).columns)
    df_resampled['target'] = y_smote_resampled
    
    st.write("### Target Distribution After SMOTE")
    fig, ax = plt.subplots()
    sns.countplot(x=y_smote_resampled, palette=['blue', 'red'], ax=ax)
    ax.set_title('Distribusi Kelas Setelah SMOTE')
    ax.set_xlabel('Kelas Target')
    ax.set_ylabel('Jumlah')
    st.pyplot(fig)
    
    scaler = StandardScaler()
    X_res = scaler.fit_transform(X_smote_resampled)
    
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_smote_resampled, test_size=0.2, random_state=42)
    
    models = {
        "SVM": SVC(),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }
    
    results = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results.append((model_name, accuracy))
    
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    
    st.write("### Model Accuracy Comparison")
    fig, ax = plt.subplots()
    sns.barplot(x='Model', y='Accuracy', data=results_df, ax=ax)
    ax.set_title('Model Accuracy Comparison')
    st.pyplot(fig)
    
    # Confusion Matrix for Decision Tree
    dt = models["Decision Tree"]
    y_pred_dt = dt.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_dt)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    ax.set_title('Confusion Matrix - Decision Tree')
    st.pyplot(fig)
