# Import library
from nbformat import write
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import pickle

# thu vien Tokenizer Viet
from pyvi import ViTokenizer, ViPosTagger
from underthesea import word_tokenize, pos_tag, sent_tokenize
from wordcloud import WordCloud

# data pre-processing libraries
import regex
import demoji
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


######################################################################################################
raw_data = pd.read_csv('data/data_Foody.csv')


# GUI
st.sidebar.markdown("<h1 style='text-align: left; color: Black;'>CATALOG</h1>", unsafe_allow_html=True)
menu = ["Introduction","Summary about Projects", 'Sentiment Analysis', "Conclusion and Next steps"]
choice = st.sidebar.radio("Choose one of objects below", menu)

st.write("<h1 style='text-align: left; color: Red; font-size:40px'>FOOD SENTIMENT ANALYSIS</h1>", unsafe_allow_html=True)


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

# 3. Sentiment Analysis
elif choice == 'Sentiment Analysis':

#----------------------------------------------------------------------------------------------------
    # if __name__=='__main__':
    #     model_pre = pickle.load(open('model/Project2_model_pre_processing.sav', 'rb'))
    #     model_predict = pickle.load(open('model/Project2_mb_model_prediction.sav', 'rb'))
    def data_cleaning(data):
        df = data.copy()

        # remove unnamed columns
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

        # feature "review_text"
        df_clean = df
        document = df_clean['review_text']

        # load edited emojicon
        file = open('lib/files/emojicon.txt', 'r', encoding="utf8")
        emoji_lst = file.read().split('\n')
        file.close()

        emoji_dict = {}
        lst_key = []
        lst_value = []
        for line in emoji_lst:
            key, value = line.split('\t')
            lst_key.append(key)
            lst_value.append(value)
            emoji_dict[key] = str(value)

        # load chosen words
        file = open('lib/files/chosen_words_full.txt', 'r', encoding="utf8")
        lst_chosen_words = file.read().split('\n')
        file.close()

        lst_spec_symbol = ['~','`','!','@','#','$','%','^','&','*','(',')','-','=','+',   '[','{','}','}','\\','|',';',':','"',     ',','<','.','>','/','?']
        lst_find_words = ['không ','ko ','kg ','chẳng ','chả ']
        lst_replace_words = ['không_','ko_','kg_','chẳng_','chả_']

        lst_document = []
        for doc in document:
            doc = doc.lower()

            # TEXT CLEANING
            # replace emojicons
            for i in range(len(lst_key)):
                doc = doc.replace(lst_key[i], ' '+lst_value[i])
            for j in doc:
                if j in emoji.UNICODE_EMOJI['en']: 
                    doc = doc.replace(j, '')

            # word tokenize: sạch sẽ => sạch_sẽ
            doc = word_tokenize(doc, format='text')

            # remove special symbols
            rx = '[' + re.escape(''.join(lst_spec_symbol)) + ']' 
            doc = re.sub(rx, '', doc)

            doc = doc.replace('  ',' ')
            doc = doc.replace(' _',' ')
            doc = doc.replace('_ ',' ')

            # replace 'không ' to 'không_' co link words, v.v...
            for i in range(len(lst_find_words)):
                doc = doc.replace(lst_find_words[i], lst_replace_words[i]) 

            # remove stop_words
            lst_words = []
            for j in doc.split(' '):
                if j in lst_chosen_words: lst_words.append(j)
            doc = ' '.join(lst_words)  

            lst_document.append(doc)

        df_clean['review_text_clean'] = lst_document
        # df_clean = df_clean.dropna()

        # create "review_score_new"
        df_clean['review_score_new'] = 0
        df_clean.loc[df_clean['review_score'] >= 6.8, 'review_score_new'] = 1
        df_clean['review_score_new'] = df_clean['review_score_new'].astype('int32')

        # load encoder model
        cv = pickle.load(open('model/CountVectorizer_self_model.sav','rb'))

        # COUNTVECTORIZER
        # apply CountVectorizer
        # cv_transformed = cv.transform(df_clean['review_text_clean'].dropna())
        cv_transformed = cv.transform(df_clean['review_text_clean'])
        cv_transformed = cv_transformed.toarray()
        df_clean_cv = pd.DataFrame(cv_transformed, columns=cv.get_feature_names())
        df_clean_cv = pd.concat([df_clean.reset_index(), df_clean_cv], axis=1)

        # CREATE X, y   
        X = df_clean_cv.iloc[:,6:]
        y = df_clean_cv['review_score_new']

        return X, y

#----------------------------------------------------------------------------------------------------

    # content
    st.header('SENTIMENT ANALYSIS')

    # sidebar
    menu3_input = ['Input data','Load file']
    choice3_input = st.sidebar.selectbox('Choose the way to input data', menu3_input)

    menu3_model = ['Model 1','Model 2']
    choice3_model = st.sidebar.selectbox('Choose the model', menu3_model)


    if choice3_input =='Input data':
        # sidebar - input
        fea_restaurant = st.text_input('Restaurant name') 
        fea_review_text = st.text_input('Review text')
        

        # content
        if choice3_model == 'Model 1':
        
            # prediction - CODE HERE
            

            # show results - CODE HERE
            if st.button("Show predict results"):
                st.write('not done')
                    


        elif choice3_model == 'Model 2':

            model_predict = pickle.load(open('model/Project2_mb_model_prediction.sav', 'rb'))

            # prediction
            lst_input = [fea_restaurant, fea_review_text, 0.5]
            col_names = ['restaurant','review_text','review_score']
            input_data = pd.DataFrame(lst_input)
            input_data = input_data.T
            input_data.columns = col_names

            X_new, y_new = data_cleaning(input_data)
            yhat = model_predict.predict(X_new)

            result_df = pd.concat([input_data[['restaurant','review_text']], pd.DataFrame(yhat)], axis=1)
            result_df.columns = ['restaurant','review_text','sentiment']
            result_df.loc[result_df['sentiment']==0, 'sentiment'] = 'positive'
            result_df.loc[result_df['sentiment']==1, 'sentiment'] = 'negative'

            # show results
            if st.button("Show predict results"):
                st.write('Sentiment')
                st.dataframe(result_df)

            
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

        except Exception as failGeneral:        
            print("Fail system, please call developer...", type(failGeneral).__name__)
            print("Description:", failGeneral)
        finally:
            print("File uploaded")



        # show results
        if st.sidebar.button("Show predict results"):

            uploaded_data = pd.read_csv(uploaded_file)
            
        # content
            if choice3_model == 'Model 1':  

                # prediction - CODE HERE
                

                # download results - CODE HERE
                st.write('not done')

            elif choice3_model == 'Model 2':

                model_predict = pickle.load(open('model/Project2_mb_model_prediction.sav', 'rb'))

                # prediction
                uploaded_data['review_score'] = 0.5
                X_new, y_new = data_cleaning(uploaded_data)
                yhat = model_predict.predict(X_new)

                result_df = pd.concat([uploaded_data[['restaurant','review_text']], pd.DataFrame(yhat)], axis=1)
                result_df.columns = ['restaurant','review_text','sentiment']
                result_df.loc[result_df['sentiment']==0, 'sentiment'] = 'positive'
                result_df.loc[result_df['sentiment']==1, 'sentiment'] = 'negative'
                # pd.set_option('display.max_colwidth', None)

                # show results
                st.write("<h5 style='text-align: left; color: Black;'>Input information:</h5>", unsafe_allow_html=True)
                st.write("1. Model name: Model 2")
                st.markdown("2. Input file name: " + str(uploaded_file.name))
                st.markdown("3. Number of distinct rows: " + str(uploaded_data.shape[0]))
                st.markdown("4. Number of columns: " + str(uploaded_data.shape[1]-1))
                st.write("<h5 style='text-align: left; color: Black;'>Prediction results:</h5>", unsafe_allow_html=True)
                st.dataframe(result_df)
                download_results = result_df.to_csv().encode('utf-8')

                # download results
                st.download_button(label="Download data as CSV",
                                    data=download_results,
                                    file_name='predict_results_model2.csv',
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

    




# 4. Conclusion and Next steps
elif choice == 'Conclusion and Next steps':

    st.header('CONCLUSION AND NEXT STEPS')

    



