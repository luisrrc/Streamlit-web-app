import streamlit as st
import pandas as pd
import numpy as np
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io

#------------------------------------------------------

# Page layout


#------------------------------------------------------
def app():

    st.title('The Classification Algorithm Comparison App')

    
# Building the model

    def build_model(df):
        X = df.iloc[:,:-1]
        Y = df.iloc[:,-1]

        st.markdown('**1.2. Dataset dimension**')
        st.write('X')
        st.info(X.shape)
        st.write('Y')
        st.info(Y.shape)

        st.markdown('**1.3. Variable details**:')
        st.write('X variable (first 30 are shown)')
        st.info(list(X.columns[:30]))
        st.write('Y variable')
        st.info(Y.name)

    # Building the lazy model

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = split_size, random_state = seed_number)
        clf = LazyClassifier(verbose = 0, ignore_warnings = True, custom_metric = None)
        models_train, predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
        models_test, predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)

        st.subheader('2. Table of Model Performance')

        st.write('Training set')
        st.write(predictions_train)
        st.markdown(filedownload(predictions_train, 'training.csv'), unsafe_allow_html = True)

        st.write('Test set')
        st.write(predictions_test)
        st.markdown(filedownload(predictions_test, 'test.csv'), unsafe_allow_html= True)

        st.subheader('3. Plot of Model Performance')

        with st.markdown('**Accuracy training set**'):

        # Tall

            plt.figure(figsize = (5, 10))
            sns.set_theme(style = "whitegrid")
            ax = sns.barplot(y = models_train.index, x = "Accuracy", data = models_train)
        st.markdown(imagedownload(plt, 'plot-acc-tall.pdf'), unsafe_allow_html = True)

        # Wide

        plt.figure(figsize = (10, 5))
        sns.set_theme(style = "whitegrid")
        ax = sns.barplot(x = models_train.index, y = "Accuracy", data = models_train)
        plt.xticks(rotation = 90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-acc-wide.pdf'), unsafe_allow_html = True)


        with st.markdown('**Accuracy test set**'):

        # Tall

            plt.figure(figsize = (5, 10))
            sns.set_theme(style = "whitegrid")
            ax = sns.barplot(y = models_test.index, x = "Accuracy", data = models_test)
        st.markdown(imagedownload(plt, 'plot-acc-tall.pdf'), unsafe_allow_html = True)

        # Wide

        plt.figure(figsize = (10, 5))
        sns.set_theme(style = "whitegrid")
        ax = sns.barplot(x = models_test.index, y = "Accuracy", data = models_test)
        plt.xticks(rotation = 90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-acc-wide.pdf'), unsafe_allow_html = True)

# Donwload CSV data

    def filedownload(df, filename):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href = "data : file / csv; base64, {b64}" download = {filename} > Download {filename} File</a>'
        return href

    def imagedownload(plt, filename):
        s = io.BytesIO()
        plt.savefig(s, format = 'pdf', bbox_inches = 'tight')
        plt.close()
        b64 = base64.b64encode(s.getvalue()).decode()
        href = f'<a href = "data : image / png; base, {b64}" download = {filename} > Download {filename} File</a>'
        return href

#-----------------------------------------------------------------------

    st.write("""
    # The Classification Algorithm Comparison App
    Developed by: [Luis Ruiz]
    """)

#-----------------------------------------------------------------------

    # Sidebar - Collects user input features into dataframe
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type = ["csv"])
        st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/luisrrc/Tarjeta-de-Credito-Default/main/UCI_Credit_Card.csv)
    """)

    # Sidebar - Specify parameter settings
    with st.sidebar.header('2. Set Parameters'):
        split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)
        seed_number = st.sidebar.slider('Set the random seed number', 1, 100, 42, 1)

#------------------------------------------------------------------------

    # Main Panel

    # Display the Dataset
    st.subheader('1. Dataset')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown('**1.1. Glimpse of dataset**')
        st.write(df)
        build_model(df)
    else:
        st.info('Awaiting for CSV file to be uploaded.')
        if st.button('Press to use Example Dataset'):
            # Diabetes dataset
            breast_cancer = load_breast_cancer()
            X = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
            Y = pd.Series(breast_cancer.target, name = 'response')
            df = pd.concat( [X,Y], axis=1 )

            st.markdown('The Diabetes dataset is used as the example.')
            st.write(df.head(5))

            build_model(df)