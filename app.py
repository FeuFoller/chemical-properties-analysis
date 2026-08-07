import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/random_forest.pkl")

st.title("🧪 Chemical Solubility Predictor")

st.write(
    "Predict aqueous solubility from molecular descriptors."
)
MolWt = st.number_input("Molecular Weight", value=250.0)

MolLogP = st.number_input("LogP", value=2.0)

TPSA = st.number_input("TPSA", value=50.0)

HeavyAtomCount = st.number_input("Heavy Atom Count", value=20)

NumHAcceptors = st.number_input("Hydrogen Bond Acceptors", value=3)

NumHDonors = st.number_input("Hydrogen Bond Donors", value=1)

NumRotatableBonds = st.number_input("Rotatable Bonds", value=4)

MolMR = st.number_input("Molecular Refractivity", value=60.0)

if st.button("Predict"):

    sample = pd.DataFrame({
        "MolWt":[MolWt],
        "MolLogP":[MolLogP],
        "TPSA":[TPSA],
        "HeavyAtomCount":[HeavyAtomCount],
        "NumHAcceptors":[NumHAcceptors],
        "NumHDonors":[NumHDonors],
        "NumRotatableBonds":[NumRotatableBonds],
        "MolMR":[MolMR]
    })

    prediction = model.predict(sample)

    st.success(
        f"Predicted Solubility: {prediction[0]:.3f}"
    )

with st.sidebar:
    st.header("About")
    st.write(
        """
        Predict aqueous solubility using
        a Random Forest model trained on
        molecular descriptors.
        """
    )
st.markdown("---")

st.caption(
    "Built with Python, scikit-learn and Streamlit."
)

