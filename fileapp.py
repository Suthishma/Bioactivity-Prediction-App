import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
from sklearn.preprocessing import StandardScaler  # Import necessary preprocessing modules


# Molecular descriptor calculator
def desc_calc():

    # Performs the descriptor calculation
    bashCommand = f"java -Xms2G -Xmx2G -Djava.awt.headless=true -jar C:/Users/HP/AppData/Local/Programs/Python/Python312/Scripts/Drug_ML_Project/PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes C:/Users/HP/AppData/Local/Programs/Python/Python312/Scripts/Drug_ML_Project/PaDEL-Descriptor/PubchemFingerprinter.xml -dir C:/Users/HP/AppData/Local/Programs/Python/Python312/Scripts/Drug_ML_Project -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    os.remove('molecule.smi')

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data):
    try:
        # Load the trained model
        with open('C:/Users/HP/AppData/Local/Programs/Python/Python312/Scripts/Drug_ML_Project/ML_best_acetyl_best_model_final.pkl', 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Apply preprocessing to match the features used during model training
        # scaler = StandardScaler()
        # input_data_scaled = scaler.fit_transform(input_data)
        
        # Adjust the input data to match the expected features if needed
        
        # Make predictions using the loaded model
        prediction = loaded_model.predict(input_data)
        
        # Display prediction results
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pIC50')
        molecule_name = pd.Series(load_data[1], name='molecule_name')
        df = pd.concat([molecule_name, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error occurred during prediction: {e}")


# Logo image
image = Image.open('C:/Users/HP/AppData/Local/Programs/Python/Python312/Scripts/Drug_ML_Project/logo.png')

st.image(image, use_column_width=True)

# Page title
st.markdown("""
# Bioactivity Prediction App (Acetylcholinesterase)

This app allows you to predict the bioactivity towards inhibiting the `Acetylcholinesterase` enzyme. `Acetylcholinesterase` is a drug target for Alzheimer's disease.
""")

# Sidebar
with st.sidebar.header('1. Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](example_arotxt)
""")

if st.sidebar.button('Predict'):
    uploaded_file_path = 'C:/Users/HP/AppData/Local/Programs/Python/Python312/Scripts/Drug_ML_Project/' + uploaded_file.name
    load_data = pd.read_table(uploaded_file_path, sep=' ', header=None)
    load_data.to_csv('molecule.smi', sep='\t', header=False, index=False)

    st.header('**Original input data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        desc_calc()

    # Read in calculated descriptors and display the dataframe
    st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('descriptors_output.csv')
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header('**Subset of descriptors from previously built models**')
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc_subset)
else:
    st.info('Upload input data in the sidebar to start!')
