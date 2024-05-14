import pandas as pd

import streamlit as st

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(layout="wide")

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
            
                .reportview-container {
                     margin-top: -2em;
                }
                #MainMenu {visibility: hidden;}
                    .stDeployButton {display:none;}
                    footer {visibility: hidden;}
                    #stDecoration {display:none;}
                
                

        </style>
        """, unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(['Test data', 'Train data', 'Validation'])


data = None

with tab1:
    cont1 = st.container
    c1,c2,c3,c4 = st.columns((0.7,1,1,2))
    c1.markdown(
            "<div style='text-align: center; padding-top:10px; font-size:24px; font-weight:bold;'>Upload data</div>",
            unsafe_allow_html=True)
    upload = c1.file_uploader(' ')
    if upload is not None:
        data = pd.read_csv(upload)
        data = data[['A','B','C','G']]
    

    c2.markdown(
            "<div style='text-align: center; padding-top:10px; font-size:24px; font-weight:bold;'>Describe</div>",
            unsafe_allow_html=True)
    if data is not None:
        c2.dataframe(data=data.describe(), use_container_width=True)
    
    c3.markdown(
            "<div style='text-align: center; padding-top:10px; font-size:24px; font-weight:bold;'>Head</div>",
            unsafe_allow_html=True)
    if data is not None:
     c3.dataframe(data=data.head(), use_container_width=True)

    c4.markdown(
            "<div style='text-align: center; padding-top:10px; font-size:24px; font-weight:bold;'>Graph</div>",
            unsafe_allow_html=True)
    exp = c4.expander('Explore', False)
    if data is not None:
        exp.line_chart(data=data)

import Classifier as ml

with tab2:
    cont2 = st.container
    c1,c2,c3 = st.columns((1,1,1))

    c1.subheader('Dataset')
    c1.write('Default data: *RING_V2.csv*')

    upload = c1.file_uploader('Upload new data')
    if upload is not None:
            ml.default_data = pd.read_csv(upload)

    c2.subheader('Explore data')
    exp = c2.expander('Methods', False)
    exp.write("Describe")
    exp.dataframe(data=ml.default_data.describe(), use_container_width=True)
    exp.write("Head")
    exp.dataframe(data=ml.default_data.head(), use_container_width=True)

    c3.subheader('Data after editing')

    X_train, X_test, y_train, y_test = ml.dataPre(ml.default_data)

    exp = c3.expander('Data split', False)
    with exp:
        ic1, ic2 = st.columns((2,1))
        ic1.write('X Train')
        ic2.write('Y Train')
        ic1.dataframe(data=X_train, use_container_width=True)
        ic2.dataframe(data=y_train, use_container_width=True)
        ic1.write('X Test')
        ic2.write('Y Test')
        ic1.dataframe(data=X_test, use_container_width=True)
        ic2.dataframe(data=y_test, use_container_width=True)

    line = st.write("-------------------------------------------------------------------------------------------------------------------------------------------")

    max_depth=None
    min_samples_leaf=1 
    min_samples_split=2
    n_estimators = 300

    max_depth = st.slider('max_depth', min_value = None, max_value = 20, value = None)
    min_samples_leaf = st.slider('min_samples_leaf', min_value = 1, max_value = 10, value = 1)
    min_samples_split = st.slider('min_samples_split', min_value = 2, max_value = 10, value = 2)
    n_estimators = st.slider('n_estimators', min_value = 100, max_value = 1000, value = 300, step=10)

    train_button = st.button('Train on current data', use_container_width=True)

    if(max_depth == 0):
        max_depth = None
    
    if(train_button):
        ml.Tr(max_depth, min_samples_leaf, min_samples_split, n_estimators)
        accuracy, f1 =  ml.training(X_train, X_test, y_train, y_test)
        label = st.subheader('Training results')
        st.write(f'Accuracy:{accuracy}')
        st.write(f'F1:{f1}')
    
with tab3:
     
    test_button = st.button('Validate model', use_container_width=True)
    
    c1,c2 = st.columns((1,1))  



    c1.markdown(
            "<div style='text-align: center; padding-top:10px; font-size:24px; font-weight:bold;'>Results</div>",
            unsafe_allow_html=True)
    if (test_button):
        accuracy, f1, _, _ = ml.validation(data)
        c1.write(f'Accuracy {accuracy}')
        c1.write(f'F1 {f1}')
    
    c2.markdown(
            "<div style='text-align: center; padding-top:10px; font-size:24px; font-weight:bold;'>Visualization</div>",
            unsafe_allow_html=True)
    if (test_button):
        _,_, y_exam, y_pred = ml.validation(data)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        cm = confusion_matrix(y_exam, y_pred)

        cm_normalized = cm / cm.max()

        plt.figure(figsize=(10, 8))
        plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Normalized Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_pred)))
        plt.xticks(tick_marks, np.unique(y_pred))
        plt.yticks(tick_marks, np.unique(y_pred))

        thresh = cm_normalized.max() / 2.
        for i, j in np.ndindex(cm_normalized.shape):
            plt.text(j, i, format(cm_normalized[i, j], '.2f'),
                    horizontalalignment="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        c2.pyplot(plt.show(), use_container_width=True)
                
   


