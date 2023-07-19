import streamlit as st
import numpy as np
import pandas as pd

st.title("Adaline Demo")

type = st.sidebar.radio(
    "Select the type of Input and Target:",
    ('Binary', 'Bipolar'))

if type == 'Binary':
    T = 1
    F = 0
else:
    T = 1
    F = -1



gate = st.sidebar.radio(
    "Select the logical gate:",
    ('AND', 'OR'))

if gate == 'AND':
    data = {
    'x1':[T,T,F,F],
    'x2':[T,F,T,F],
    't': [T,F,F,F]
    }
else:
    data = {
    'x1':[T,T,F,F],
    'x2':[T,F,T,F],
    't': [T,T,T,F]
    }


data_df = pd.DataFrame(data)

def adaline(x1, x2, t, epochs, loss_threshold=0):
    data ={
        'x1' : [],
        'x2' : [],
        't'  : [],
        'Yin': [],
        'error': [],
        'dw1': [],
        'dw2': [],
        'dbias': [],
        'w1' : [],
        'w2' : [],
        'bias'  : [],
        'errorSq' : [],
        'epoch' : []
        
    }
    w1 = w2 = 0.1
    bias = 0.1
    rate = 0.1
    n = len(x1)
    for i in range(epochs):
        total_error = 0
        for j in range(n):
            data['x1'].append(x1[j])
            data['x2'].append(x2[j])
            data['t'].append(t[j])
            Yin = w1 * x1[j] + w2 * x2[j] + bias
            data['Yin'].append(Yin)
            error = t[j] - Yin
            data['error'].append(error)
            errorSq = error**2
            data['errorSq'].append(errorSq)
            
            w1_c = rate * error * x1[j]
            w2_c = rate * error * x2[j]
            bias_c = rate * error
            
            data['dw1'].append(w1_c)
            data['dw2'].append(w2_c)
            data['dbias'].append(bias_c)
            
            w1 = w1 + w1_c
            w2 = w2 + w2_c
            bias = bias + bias_c
            
            data['w1'].append(w1)
            data['w2'].append(w2)
            data['bias'].append(w1)
            data['epoch'].append(i)
            total_error = total_error + errorSq
            
            
            #print(error)
        print(total_error)
        if total_error<=loss_threshold:
            break
    #print(data)
    df = pd.DataFrame(data)
    print(df)
    return df

st.write("""## DataSet""")
st.table(data)

ep = st.sidebar.slider('Epochs',1,1000,5)

te = st.sidebar.slider('Error_Threshold',0.0,10.0,2.0)


ada_df = adaline(data_df.x1,data_df.x2,data_df.t,ep,te)

#st.write(ada_df)
n_ep = ada_df.epoch.nunique()

for i in range(n_ep):
    
    dataframe = ada_df[ada_df['epoch']==i]
    dataframe['epoch'] = dataframe['epoch']+1
    st.write(f"""
    ## Epoch {i+1}
    """)
    st.table(dataframe)
    st.write('Total Error',dataframe['errorSq'].sum())
    st.write("")
    st.write("")