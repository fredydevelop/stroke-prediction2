#Importing the dependencies
import numpy as np
import pandas as pd
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import streamlit as st
from streamlit_option_menu import option_menu
import base64
import pickle as pk
import joblib
# import option_menu




#configuring the page setup
st.set_page_config(page_title='Stroke prediction system',layout='centered')

with st.sidebar:
    st.title("Home Page")
    selection=option_menu(menu_title="Main Menu",options=["Single Prediction","Multi Prediction"],icons=["cast","book","cast"],menu_icon="house",default_index=0)


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your prediction</a>'
    return href


#single prediction function
def stroke_detect(givendata):
    # loaded_model = joblib.load("heartcheck.sav")

    loaded_model=pk.load(open("stroke2.sav", "rb"))
    input_data_as_numpy_array = np.asarray(givendata)# changing the input_data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the array as we are predicting for one instance
    std_scaler_loaded=pk.load(open("std_scaler.pkl", "rb"))
    std_X_resample=std_scaler_loaded.transform(input_data_reshaped)
    prediction = loaded_model.predict(std_X_resample)
    if prediction==1:
      return "Stroke present"
    else:
      return "No Stroke Issues Present"
    

#main function handling the input
def main():
    st.header("Stroke Detection and Predictive System")
    
    #getting user input
    
    age = st.slider('Patient age', 0, 250, key="ageslide")
    st.write("Patient's is :", age, 'years old')

    option1 = st.selectbox('sex',("",'Male' ,'Female'),key="gender")
    if (option1=='Male'):
        sex=1
    else:
        sex=0

    option4 = st.selectbox('hypertension',("","hypertensive","Not hypertensive"),key="hyper")
    if (option4=="hypertensive"):
        hyperten=1
    else:
        hyperten=0

    option5 = st.selectbox('heart_disease',("","has heart disease", "No heart disease"),key="hertdis")
    if option5== "has heart disease":
        heartDis=1
    else:
        heartDis=0


    option14 = st.selectbox("Ever Married ? ",("","Yes", "No"),key="married_")
    if option14== "Yes":
        married=1
    else:
        married=0

    option6 = st.selectbox("Work Type ",("","children", "Govt_jov", "Never_worked", "Private" "Self-employed"),key="worktype")
    if option6== "children":
        work_type=4
    elif option6=="Private":
        work_type=2
    elif option6=="Self-employed":
        work_type=3
    elif option6=="Govt_jov":
        work_type=0
    else:
        work_type=1


    option7 = st.selectbox('Residence Type',("",'Rural',"Urban"),key="residentType")
    if (option7=='Rural'):
        resident=0
    else:
        resident=1


    glucose = st.number_input('Average Glucose level',key="avgGlucose")
    st.write('The average glucose level is ', glucose)

    st.write("\n")
    st.write("\n")

    bmi = st.number_input('Body Mass Index',key="b_m_i")
    st.write('Body Mass Index is ', bmi)


    smoking_status = st.selectbox('Smokin Status',("","formerly smoked", "never smoked", "smokes"),key="smoking")
    if (smoking_status=='formerly smoked'):
        smoking_stat=1
    
    elif smoking_status=="never smoked":
        smoking_stat=0

    else:
        smoking_stat=2

    st.write("\n")
    st.write("\n")





    detectionResult = ''#for displaying result
    
    # creating a button for Prediction
    if age!="" and option1!="" and option4!="" and option5!="" and option6!="" and option7 !="" and option14 !="" and glucose!="" and bmi!="" and smoking_status!="" and st.button('Predict'):
        detectionResult = stroke_detect([age,sex,hyperten,heartDis, married,work_type, resident, glucose, bmi, smoking_stat])
        st.success(detectionResult)


def multi(input_data):
    loaded_model=pk.load(open("stroke2.sav", "rb"))
    dfinput = pd.read_csv(input_data)
    dfinput.drop(dfinput.columns[0],axis=1,inplace=True)
    # if "Unname" or "id" in dfinput.iloc[1:]:
    #     dfinput.drop("Unname",axis=1,inplace=True)
    #     dfinput.drop("id",axis=1,inplace=True)


    if "stroke" in dfinput.iloc[1:]:
        dfinput.drop("stroke",axis=1,inplace=True)
    # dfinput=dfinput.reset_index(drop=True)

    st.header('A view of the uploaded dataset')
    st.markdown('')
    st.dataframe(dfinput)

    dfinput=dfinput.values
    std_scaler_loaded=pk.load(open("std_scaler.pkl", "rb"))
    std_dfinput=std_scaler_loaded.transform(dfinput)
    
    
    predict=st.button("predict")


    if predict:
        prediction = loaded_model.predict(std_dfinput)
        interchange=[]
        for i in prediction:
            if i==1:
                newi="Stroke detected"
                interchange.append(newi)
            elif i==0:
                newi="No Stroke issues present"
                interchange.append(newi)
            
        st.subheader('All the predictions')
        prediction_output = pd.Series(interchange, name='Heart attack prediction results')
        prediction_id = pd.Series(np.arange(len(interchange)),name="Patient_ID")
        dfresult = pd.concat([prediction_id, prediction_output], axis=1)
        st.dataframe(dfresult)
        st.markdown(filedownload(dfresult), unsafe_allow_html=True)
        

if selection == "Single Prediction":
    main()

if selection == "Multi Prediction":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #---------------------------------#
    # Prediction
    #--------------------------------
    #---------------------------------#
    # Sidebar - Collects user input features into dataframe
    st.header('Upload your csv file here')
    uploaded_file = st.file_uploader("", type=["csv","xls"])
    #--------------Visualization-------------------#
    # Main panel
    
    # Displays the dataset
    if uploaded_file is not None:
        #load_data = pd.read_table(uploaded_file).
        multi(uploaded_file)
    else:
        st.info('Upload your dataset !!')
    