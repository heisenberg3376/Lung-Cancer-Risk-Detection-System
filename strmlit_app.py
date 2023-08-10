import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import xgboost
import numpy as np
import webbrowser

model = pickle.load(open('XGB.pkl','rb'))


st.set_page_config('wide')

selected = option_menu('Main Menu',['Home','Info','About me','Github'],icons=['house','book','person', 'gear',],orientation='horizontal')
if(selected=='Home'):
    st.markdown('# Lung Cancer Risk Detection System')
    st.write('Select your symptoms and click on ')
    age = st.slider('Age',min_value=1,max_value=99,value=40)
    air_pollution = st.selectbox('Air Pollution Level',np.arange(1,9))
    alcohol_use = st.selectbox('Alcohol Use',np.arange(1,9))
    dust_allergy = st.selectbox('Dust Allergy',np.arange(1,9))
    occ_hazards = st.selectbox('Occupational Hazards',np.arange(1,9))
    gen_risk = st.selectbox('Genetic Risk',np.arange(1,9))
    cr_lung_disease = st.selectbox('Chronic Lung Disease',np.arange(1,9))
    bal_diet = st.selectbox('Balanced Diet',np.arange(1,9))

    obesity = st.selectbox('Obesity',np.arange(1,9))
    smoking = st.selectbox('Smoking',np.arange(1,9))
    pass_smoking = st.selectbox('Passive Smoking',np.arange(1,9))
    chest_pain = st.selectbox('Chest Pain',np.arange(1,9))
    coughing_bld = st.selectbox('Coughing of Blood',np.arange(1,9))
    fatigue = st.selectbox('Fatigue',np.arange(1,9))
    dry_cough = st.selectbox('Dry Cough',np.arange(1,9))

    features = np.array([age, air_pollution, alcohol_use, dust_allergy,
                        occ_hazards, gen_risk, cr_lung_disease, bal_diet,
                        obesity, smoking, pass_smoking, chest_pain,coughing_bld,
                        fatigue, dry_cough])
    preds = model.predict(features.reshape(1,-1))
    if(st.button('Proceed')):
        if(preds == 1):
            st.markdown('## The Patient has Less risk of Lung Cancer')
        if(preds == 2):
            st.markdown('## The Patient has Moderate risk of Lung Cancer')
        if(preds == 0):
            st.markdown('## The Patient has High risk of Lung Cancer')

if(selected=='Info'):
    st.write("<h1 style='color:white;font-family:monospace' align=center>More Information</h1><br><br><p style='color:white;font-family:monospace;font-size:20px'>&nbsp&nbsp&nbsp&nbsp Lung cancer is the leading cause of cancer death worldwide, accounting for 1.59 million deaths in 2018. The majority of lung cancer cases are attributed to smoking, but exposure to air pollution is also a risk factor. A new study has found that air pollution may be linked to an increased risk of lung cancer, even in nonsmokers.<br>&nbsp&nbsp&nbsp&nbspThe study, which was published in the journal Nature Medicine, looked at data from over 462,000 people in China who were followed for an average of six years. The participants were divided into two groups: those who lived in areas with high levels of air pollution and those who lived in areas with low levels of air pollution.The researchers found that the people in the high-pollution group were more likely to develop lung cancer than those in the low-pollution group.<br>&nbsp&nbsp&nbsp&nbsp They also found that the risk was higher in nonsmokers than smokers, and that the risk increased with age.While this study does not prove that air pollution causes lung cancer, it does suggest that there may be a link between the two. More research is needed to confirm these findings and to determine what effect different types and levels of air pollution may have on lung cancer risk</p><br><br><br><h1 style='color:white;font-family:monospace' align=center>Info About the Data</h2><br><p style='color:white;font-family:monospace;font-size:20px'>This dataset contains information on patients with lung cancer, including their age, air pollution exposure, alcohol use, dust allergy, occupational hazards, genetic risk, chronic lung disease, balanced diet, obesity, smoking status, passive smoker status, chest pain, coughing of blood, fatigue levels , dry coughs . By analyzing this data we can gain insight into what causes lung cancer.<br><br>Age: The age of the patient. (Numeric)<br><br>Gender: The gender of the patient. (Categorical)<br><br>Air Pollution: The level of air pollution exposure of the patient. (Categorical)<br><br>Alcohol use: The level of alcohol use of the patient. (Categorical)<br><br>Dust Allergy: The level of dust allergy of the patient. (Categorical)<br><br>OccuPational Hazards: The level of occupational hazards of the patient. (Categorical)<br><br>Genetic Risk: The level of genetic risk of the patient. (Categorical)<br><br>chronic Lung Disease: The level of chronic lung disease of the patient. (Categorical)<br><br>Balanced Diet: The level of balanced diet of the patient. (Categorical)<br><br>Obesity: The level of obesity of the patient. (Categorical)<br><br>Smoking: The level of smoking of the patient. (Categorical)<br><br>Passive Smoker: The level of passive smoker of the patient. (Categorical)<br><br>Chest Pain: The level of chest pain of the patient. (Categorical)<br><br>Coughing of Blood: The level of coughing of blood of the patient. (Categorical)<br><br>Fatigue: The level of fatigue of the patient. (Categorical)<br><br>Weight Loss: The level of weight loss of the patient. (Categorical)<br><br>Shortness of Breath: The level of shortness of breath of the patient. (Categorical)<br><br>Wheezing: The level of wheezing of the patient. (Categorical)<br><br>Swallowing Difficulty: The level of swallowing difficulty of the patient. (Categorical)<br><br>Clubbing of Finger Nails: The level of clubbing of finger nails of the patient. (Categorical)<br><br><br>Levels:<br>1 : No<br>2 : Very Low<br>3 : Low<br>4 : Average<br>5 : Above Average<br>6 : High<br>7 : Very High<br>8 : Dangerous<br></p>",unsafe_allow_html=True)
    st.write("<a style='color:white;font-family:monospace;font-size:20px;href=https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link'>This is the Dataset</a>",unsafe_allow_html=True)
    st.write("<a style='color:white;font-family:monospace;font-size:20px;href=https://www.github.com/heisenberg3376/Lung-Cancer-Risk-Detection-System'/>This is the Github Repo</a>",unsafe_allow_html=True)

if(selected=='About me'):
    url = 'https://heisenberg3376.github.io/PhanendraPortfolio.github.io/'
    webbrowser.open_new_tab(url)
    st.write("<a style='color:white;font-family:monospace;font-size:20px;href=https://heisenberg3376.github.io/PhanendraPortfolio.github.io/'>My Portfolio</a>",unsafe_allow_html=True)


if(selected=='Github'):
    st.write("<a style='color:white;font-family:monospace;font-size:20px;href=https://www.github.com/heisenberg3376/>Github Profile</a>",unsafe_allow_html=True)


    


    
  
