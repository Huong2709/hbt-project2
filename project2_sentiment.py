# Import library
from nbformat import write
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

# thu vien Tokenizer Viet
from pyvi import ViTokenizer, ViPosTagger
from wordcloud import WordCloud

# data pre-processing libraries
import emoji
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split  
# model
from sklearn.naive_bayes import MultinomialNB, BernoulliNB


# evaluation libraries
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn import metrics
# function libraries
import sys
import os
import re

import import_ipynb
from lib.function_lib import *

raw_data = pd.read_csv('data/data_Foody.csv')


# GUI
st.sidebar.markdown("<h1 style='text-align: left; color: Black;'>CATALOG</h1>", unsafe_allow_html=True)
menu = ["Introduction","Summary about Projects", 'Sentiment Analysis', "Conclusion and Next steps"]
choice = st.sidebar.radio("Choose one of objects below", menu)

st.write("<h1 style='text-align: left; color: Red; font-size:40px'>AVOCADO HASS PRICE PREDICTION</h1>", unsafe_allow_html=True)


# 1. Introduction
if choice == 'Introduction':
    st.header('INTRODUCTION')


# 2. Summary about Projects
elif choice == 'Summary about Projects':
    st.header('SUMMARY ABOUT PROJECTS')
    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Business Understanding</h3>", unsafe_allow_html=True)
    st.write("**Current status:**")


    st.write("**Objective/Problem:**")


    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Data Understanding/Acquire</h3>", unsafe_allow_html=True)

    st.image('material/EDA_Scatter_text_image.jpg')

    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Data Preparation/Acquire</h3>", unsafe_allow_html=True)

   
    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Modeling and Evaluation</h3>", unsafe_allow_html=True)
    

    st.write("<h3 style='text-align: left; color: Black; font-size:20px'>Suggestion</h3>", unsafe_allow_html=True)
    


#####################################################################################################

elif choice == 'Sentiment Analysis':

#----------------------------------------------------------------------------------------------------
    # 3.1. pre_processing
    # if __name__=='__main__':
    #     model_pre = pickle.load(open('model/Problem1_model_pre_processing.sav', 'rb'))
    #     model_scaler = pickle.load(open('model/Problem1_model_standardizing.sav', 'rb'))
        # model_ex = pickle.load(open('model/Problem1_ex_model.sav', 'rb'))
        # model_rf = pickle.load(open('model/Problem1_rf_model.sav', 'rb'))
        # model_bg = pickle.load(open('model/Problem1_bg_model.sav', 'rb'))
        # model_kn = pickle.load(open('model/Problem1_kn_model.sav', 'rb'))
        # model_ann = load_model('model/Problem1_ANN_model.h5')



    if __name__=='__main__':
        model_pre = pickle.load(open('model/Project2_model_pre_processing.sav', 'rb'))
        model_predict = pickle.load(open('model/Project2_mb_model_prediction.sav', 'rb'))




#----------------------------------------------------------------------------------------------------

    # content
    st.header('SENTIMENT ANALYSIS')

    # sidebar
    menu3_input = ['Input data','Load file']
    choice3_input = st.sidebar.selectbox('Choose the way to input data', menu3_input)

    menu3_model = ['ExtraTreesRegressor','RandomForestRefressor']
    choice3_model = st.sidebar.selectbox('Choose the model', menu3_model)


    if choice3_input =='Input data':
        # sidebar - input
        fea_type = st.sidebar.radio("Type",['conventional','organic'])
        fea_region = st.sidebar.selectbox("Region", ['region1','region2'])
        fea_PLU_4046 = st.sidebar.number_input("Volume of PLU 4046", value = 1.00, format="%.2f", step=0.01)
        fea_PLU_4225 = st.sidebar.number_input("Volume of PLU 4225", value = 1.00, format="%.2f", step=0.01)
        fea_PLU_4770 = st.sidebar.number_input("Volume of PLU 4770", value = 1.00, format="%.2f", step=0.01)
        fea_Total_Volume = st.sidebar.number_input("Total volume", value = 1.00, format="%.2f", step=0.01)
        fea_Small_Bags = st.sidebar.number_input("Number of Small bags", value = 1.00, format="%.2f", step=0.01)
        fea_Large_Bags = st.sidebar.number_input("Number of Large bags", value = 1.00, format="%.2f", step=0.01)
        fea_XLarge_Bags = st.sidebar.number_input("Number of XLarge bags", value = 1.00, format="%.2f", step=0.01)
        fea_Total_Bags = st.sidebar.number_input("Total bags", value = 1.00, format="%.2f", step=0.01)

        # show results
        if st.sidebar.button("Show predict results"):
  

        # content
            if choice3_model == 'ExtraTreesRegressor':
            
                # prediction
             

                # show results
                st.write('not done')
                       


            elif choice3_model == 'RandomForestRefressor':

                # prediction


                # show results

                st.write('not done')


            
        else:
            st.write("<h5 style='text-align: left; color: Black;'>Guidance:</h5>", unsafe_allow_html=True)
            st.write("To use specific models to predict sentiment, please follow these steps and get the predict results.")
            st.write("**Step 1: Choose the way to input data: input a specific features directlly or load a csv file**")
            

            st.write("**Step 2: Input features information**")

            st.write('2.1. If you want to input features one by one directly:')            
            
            st.write("2.2. If you want load a whole csv file, choose type of model and click 'Browse files' to load csv file. Download a sample of the csv file if needed.")    

            st.write("**Step 3: Click 'Show predict results'**")            
            
            st.write("**Step 4: Read the forecast results.**")
            
            st.write("**Step 5: In case of inputing features by loading csv file, if you want to download predict results, click 'Download data as CSV'**")
            st.image('material/Guidance_6.jpg')

            
    elif choice3_input == 'Load file':

        # sidebar
        # upload template
        upload_template = pd.read_csv('material/upload_template.csv')
        download_template = upload_template.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(label="Download template as CSV",
                            data=download_template,
                            file_name='template.csv',
                            mime='text/csv',
                            )

        # upload file
        try:
            uploaded_file = st.sidebar.file_uploader('Upload data', type = ['csv'])
            # dir_file = 'material/' + uploaded_file.name
            # st.write(uploaded_file)

        except Exception as failGeneral:        
            print("Fail system, please call developer...", type(failGeneral).__name__)
            print("Description:", failGeneral)
        finally:
            print("File uploaded")



        # show results
        if st.sidebar.button("Show predict results"):

            # uploaded_data = pd.read_csv(uploaded_file)
            uploaded_data = raw_data.head(5)


            
        # content
            if choice3_model == 'ExtraTreesRegressor':  

                # prediction
                X_new, y_new = model_pre.transform(uploaded_data)
                yhat = model_predict.predict(X_new)

                result_df = pd.concat([uploaded_data[['restaurant','review_text']], pd.DataFrame(yhat)], axis=1)
                result_df.columns = ['restaurant','review_text','sentiment']
                result_df.loc[result_df['sentiment']==0, 'sentiment'] = 'positive'
                result_df.loc[result_df['sentiment']==1, 'sentiment'] = 'negative'
                # pd.set_option('display.max_colwidth', None)

                # show results
                st.write("<h5 style='text-align: left; color: Black;'>Input information:</h5>", unsafe_allow_html=True)
                st.write("1. Model name: ExtraTreesRegressor")
                # st.markdown("2. Input file name: " + str(uploaded_file.name))
                st.markdown("3. Number of distinct rows: " + str(uploaded_data.shape[0]))
                st.markdown("4. Number of columns: " + str(uploaded_data.shape[1]))
                st.write("<h5 style='text-align: left; color: Black;'>Prediction results:</h5>", unsafe_allow_html=True)
                st.dataframe(result_df)
                download_results = result_df.to_csv().encode('utf-8')

                # download results
                st.download_button(label="Download data as CSV",
                                    data=download_results,
                                    file_name='predict_results.csv',
                                    mime='text/csv',
                                    )

            elif choice3_model == 'RandomForestRefressor':


                # prediction
                X_new, y_new = model_pre.transform(uploaded_data)
                yhat = model_predict.predict(X_new)

                result_df = pd.concat([uploaded_data[['restaurant','review_text']], pd.DataFrame(yhat)], axis=1)
                result_df.columns = ['restaurant','review_text','sentiment']
                result_df.loc[result_df['sentiment']==0, 'sentiment'] = 'positive'
                result_df.loc[result_df['sentiment']==1, 'sentiment'] = 'negative'
                # pd.set_option('display.max_colwidth', None)
                st.write(result_df)

                # show results
                st.write("<h5 style='text-align: left; color: Black;'>Input information:</h5>", unsafe_allow_html=True)
                st.write("1. Model name: ExtraTreesRegressor")
                st.markdown("2. Input file name: " + str(uploaded_file.name))
                st.markdown("3. Number of distinct rows: " + str(uploaded_data.shape[0]))
                st.markdown("4. Number of columns: " + str(uploaded_data.shape[1]))
                st.write("<h5 style='text-align: left; color: Black;'>Prediction results:</h5>", unsafe_allow_html=True)
                st.dataframe(result_df)
                download_results = result_df.to_csv().encode('utf-8')

                # download results
                st.download_button(label="Download data as CSV",
                                    data=download_results,
                                    file_name='predict_results.csv',
                                    mime='text/csv',
                                    )

              

        else:
            st.write("<h5 style='text-align: left; color: Black;'>Guidance:</h5>", unsafe_allow_html=True)
            st.write("To use specific models to predict sentiment, please follow these steps and get the predict results.")
            st.write("**Step 1: Choose the way to input data: input a specific features directlly or load a csv file**")
            

            st.write("**Step 2: Input features information**")

            st.write('2.1. If you want to input features one by one directly:')            
            
            st.write("2.2. If you want load a whole csv file, choose type of model and click 'Browse files' to load csv file. Download a sample of the csv file if needed.")    

            st.write("**Step 3: Click 'Show predict results'**")            
            
            st.write("**Step 4: Read the forecast results.**")
            
            st.write("**Step 5: In case of inputing features by loading csv file, if you want to download predict results, click 'Download data as CSV'**")
            st.image('material/Guidance_6.jpg')
       


    




# 5. Conclusion and Next steps
elif choice == 'Conclusion and Next steps':

    st.header('CONCLUSION AND NEXT STEPS')

    



