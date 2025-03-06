import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
from sklearn.preprocessing import StandardScaler  # Import necessary preprocessing modules

# Molecular descriptor calculator
def desc_calc(chembl_id, canonical_smiles):
    # Create a temporary file with the input SMILES
    with open("molecule.smi", "w") as file:
        file.write(canonical_smiles)

    # Perform the descriptor calculation
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes PaDEL-Descriptor/PubchemFingerprinter.xml -dir . -file descriptors_output.csv molecule.smi"
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
        # Define load_data variable
        load_data = pd.DataFrame({'chembl_id': [chembl_id], 'canonical_smiles': [canonical_smiles]})

        # Load the trained model
        with open('ML_best_acetyl_best_model_final.pkl', 'rb') as f:
         loaded_model = pickle.load(f)


        # Apply preprocessing to match the features used during model training
        # ... (preprocessing code)

        # Make predictions using the loaded model
        prediction = loaded_model.predict(input_data)

        # Display prediction results
        st.header('**Prediction output**')
        prediction_output = pd.Series(prediction, name='pIC50')
        df = pd.concat([load_data, prediction_output], axis=1)
        st.write(df)
        st.markdown(filedownload(df), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error occurred during prediction: {e}")


# ... (previous code)

# Logo image
image = Image.open('drugimg3-transformed.jpeg')

st.image(image, use_container_width=True)


# Page title
st.markdown("""
# Bioactivity Prediction App (Acetylcholinesterase)

This app allows you to predict the bioactivity towards inhibiting the `Acetylcholinesterase` enzyme. `Acetylcholinesterase` is a drug target for Alzheimer's disease.
""")

# Sidebar
with st.sidebar.header('1. Input Chembl ID and Canonical SMILES'):
    chembl_id = st.sidebar.text_input("Enter Chembl ID")
    canonical_smiles = st.sidebar.text_input("Enter Canonical SMILES")

if st.sidebar.button('Predict'):
    # Perform the descriptor calculation using the provided Chembl ID and Canonical SMILES
    desc_calc(chembl_id, canonical_smiles)

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
    st.info('Enter Chembl ID and Canonical SMILES in the sidebar to start!')