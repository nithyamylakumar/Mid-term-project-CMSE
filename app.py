import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
import numpy as np
import altair as alt
import hiplot as hip
from pickle import TRUE


#st.set_page_config(page_title='CMSE Project',layout="wide")
st.header("Heart disease dataset from UCI ", anchor = None)
st.subheader("Reason for the study:")
st.write("Heart disease or Cardiovascular disease (CVD) is a class of diseases that involve the **heart** or **blood vessels**. Cardiovascular diseases are the leading cause of death globally. \n Together CVD resulted in **17.9 million deaths (32.1%) in 2015**. Deaths, at a given age, from CVD are more common and have been increasing in much of the developing world, while rates have declined in most of the developed world since the 1970s.")
df = pd.read_csv("heart.csv")

st.header('Objective of the EDA')
st.markdown(
    '<p style="color:blue; font-size:22px"> The objectives of the EDA are as follows:- To get an overview of the distribution of the dataset, Check for missing numerical values, outliers or other anomalies in the dataset and Discover patterns and relationships between variables in the dataset. \n iv. Check the underlying assumptions in the dataset ', unsafe_allow_html=True
)

st.title('Dataset Information')
st.markdown(
    '<p style= "color: black; font-size:22px"> This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4. Experiments with the Cleveland database have concentrated on simply attempting to distinguish presence (values 1,2,3,4) from absence (value 0). The names and social security numbers of the patients were recently removed from the database, replaced with dummy values.', unsafe_allow_html=True
)

st.header('Sample dataset')
st.table(df.iloc[1:].head(10))

st.header('Data Description')
st.table(pd.read_csv('data_description.csv'))

# str_text_3 = "First let's check the shape of the dataset"
# st.text(str_text_3)

# img = Image.open('C:/Users/Ascen/CMSE 830/MINI PROJECT/IMAGES/shape.png')
# st.image(img, caption = 'Shape of the dataset')

st.title('Important points about the dataset')
st.markdown(
    '<p style= "color: black; font-size:22px"> 1. Sex, Fasting_BS, Ex_Induced_Ang, and Target are character variables and their data type should be object. But since they are encoded (i.e., 1 and 0 ) their data type is given as int64' , unsafe_allow_html=True
)
# Statistical properties of dataset

st.dataframe(df.describe())


# Univariate analysis of the features

tab1, tab2, tab3 = st.tabs(["Univariate Analysis", "Bivariate Analysis", "Multivariate Analysis"])

x = df.columns.tolist()
y = df.columns.tolist()

with tab1:
    st.header('Univariate analysis of the different columns')
    st.text('Our feature variable of interest is the Target variable. It denotes the presence or absence or heart disease in a patient. From the univariate analysis it is evident that there are 165 patients suffering from heart disease and 138 patients who do not have a heart disease.')
    st.title("Histogram")
    x_axis_1 = st.selectbox(label = "Select an attribute", options = x, key = 1)
    plt.figure(figsize=(6,5))
    img_1 = px.histogram(df, x = x_axis_1, color = 'Target', text_auto= True)
    st.plotly_chart(img_1)

    st.header("Distribution plot")
    x_axis_2 = st.selectbox(label = "Select an attribute", options = x, key = 2 )
    plt.figure(figsize= (6,5))
    img_2 = px.histogram(df, x= x_axis_2, color='Sex', marginal="box",hover_data=df.columns, text_auto= True)
    st.plotly_chart(img_2)

with tab2:
    st.header('Bivariate analysis of different attributes')
    x_axis_3 = st.selectbox(label = "Select an attribute", options = x, key = 3 )
    y_axis_3 = st.selectbox(label = "Select an attribute", options = y, key = 4 )
    #colour = ['Sex','Fasting_BS', 'Slope', 'Ex_Induced_Ang','Target']
    img_3 = alt.Chart(df).mark_circle(size=60).encode(
    x=x_axis_3,
    y= y_axis_3,
    color= 'Sex : O',
    tooltip=['Cholesterol', 'Fasting_BS', 'Sex']
).interactive()
    st.altair_chart(img_3)

with tab3:
    st.header('Hi Plot', anchor = None)
    img_4 = hip.Experiment.from_dataframe(df)
    img_4.to_streamlit(ret = "selected_uids", key = 'hip').display()


    st.header('Facet Plot', anchor = None)
    x_axis_4 = st.selectbox(label = "Select a column for x", options = x, key = 5 )
    y_axis_4 = st.selectbox(label = "Select a column for y", options = y, key = 6 )

    img_5 = alt.Chart(df).mark_point().encode(
        x = alt.X(x_axis_4, axis = alt.Axis(title = x_axis_4,grid = False)), y = alt.Y(y_axis_4, axis = alt.Axis(title = y_axis_4,grid = False)),
        color= alt.Color('Target'),tooltip =['Target', 'Sex', 'Max_Heart_Rate'],
        facet= alt.Facet('Target',columns=2)).interactive()
    st.altair_chart(img_5)


    st.header('Pairplot', anchor = None)
    img_6 = sns.pairplot(
    df,
    x_vars=["Age", "Resting_BP", "Cholesterol", "Max_Heart_Rate", "Oldpeak"],
    y_vars=["Age", "Resting_BP", "Cholesterol", "Max_Heart_Rate", "Oldpeak"],
    hue = 'Target',
    markers=["o", "s"],
    diag_kind="hist",
)
    st.pyplot(img_6)

