import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Streamlit Başlık
st.title("Müşteri Memnuniyeti Analizi")

# Veri Yükleme
uploaded_file = st.file_uploader("Bir CSV dosyası yükleyin", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # İlk 5 satırı gösterme
    st.write("Verinin İlk 5 Satırı:")
    st.write(df.head())

    # Veri hakkında genel bilgiler
    st.write("Veri Bilgisi:")
    st.write(df.info())

    # Eksik değerlerin kontrolü
    st.write("Eksik Değerler:")
    st.write(df.isnull().sum())

    # KNN İmputer ile eksik değerleri doldurma
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = imputer.fit_transform(df.select_dtypes(include=['float64', 'int64']))
    df[df.select_dtypes(include=['float64', 'int64']).columns] = df_imputed

    st.write("Eksik Değerler Doldurulduktan Sonra:")
    st.write(df.isnull().sum())

    # Veri görselleştirme
    st.write("Müşteri Memnuniyeti Dağılımı:")
    fig, ax = plt.subplots()
    sns.countplot(x='satisfaction', data=df, ax=ax)
    st.pyplot(fig)

    st.write("Yaş Dağılımı:")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x='Age', hue='satisfaction', multiple='stack', ax=ax)
    st.pyplot(fig)

    st.write("Seyahat Türüne Göre Memnuniyet:")
    fig, ax = plt.subplots()
    sns.countplot(x='Type of Travel', hue='satisfaction', data=df, ax=ax)
    st.pyplot(fig)

    # Veri Ön İşleme ve Model Eğitimi
    kbins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    df['Age Binned'] = kbins.fit_transform(df[['Age']])

    label_encoder = LabelEncoder()
    df['satisfaction'] = label_encoder.fit_transform(df['satisfaction'])
    df['Type of Travel'] = label_encoder.fit_transform(df['Type of Travel'])
    df['Class'] = label_encoder.fit_transform(df['Class'])
    df['Customer Type'] = df['Customer Type'].map({'Loyal Customer': 1, 'Disloyal Customer': 0})
    df['Customer Type'].fillna(0, inplace=True)
    df['Customer Type'] = df['Customer Type'].astype(int)

    df.drop(columns=['Customer Type', 'Departure/Arrival time convenient'], inplace=True)

    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
    Q1 = df[numeric_columns].quantile(0.25)
    Q3 = df[numeric_columns].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[numeric_columns] = np.where(df[numeric_columns] < lower_bound, lower_bound, df[numeric_columns])
    df[numeric_columns] = np.where(df[numeric_columns] > upper_bound, upper_bound, df[numeric_columns])

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df[numeric_columns])

    selector_variance = VarianceThreshold(threshold=0.01)
    normalized_selected_variance = selector_variance.fit_transform(normalized_data)

    selector_kbest = SelectKBest(score_func=f_classif, k=10)
    normalized_selected_kbest = selector_kbest.fit_transform(normalized_data, df['satisfaction'])

    estimator = LogisticRegression()
    selector_rfe = RFE(estimator, n_features_to_select=10, step=1)
    normalized_selected_rfe = selector_rfe.fit_transform(normalized_data, df['satisfaction'])

    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    id3_model = DecisionTreeClassifier()
    id3_model.fit(X_train, y_train)

    nb_predictions = nb_model.predict(X_test)
    knn_predictions = knn_model.predict(X_test)
    id3_predictions = id3_model.predict(X_test)

    def evaluate_model(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)
        return accuracy, precision, recall, f1, confusion

    nb_accuracy, nb_precision, nb_recall, nb_f1, nb_confusion = evaluate_model(y_test, nb_predictions)
    knn_accuracy, knn_precision, knn_recall, knn_f1, knn_confusion = evaluate_model(y_test, knn_predictions)
    id3_accuracy, id3_precision, id3_recall, id3_f1, id3_confusion = evaluate_model(y_test, id3_predictions)

    st.write("Naive Bayes Doğruluk: ", nb_accuracy)
    st.write("K-NN Doğruluk: ", knn_accuracy)
    st.write("ID3 Doğruluk: ", id3_accuracy)

    def plot_confusion_matrix(confusion_matrix, labels):
        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.2)
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot()

    st.write("Naive Bayes Karışıklık Matrisi:")
    plot_confusion_matrix(nb_confusion, ['Dissatisfied', 'Satisfied'])

    st.write("K-NN Karışıklık Matrisi:")
    plot_confusion_matrix(knn_confusion, ['Dissatisfied', 'Satisfied'])

    st.write("ID3 Karışıklık Matrisi:")
    plot_confusion_matrix(id3_confusion, ['Dissatisfied', 'Satisfied'])
