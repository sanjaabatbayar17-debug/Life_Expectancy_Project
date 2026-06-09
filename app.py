
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("WHO Life Expectancy Prediction App")
st.write("Multiple Feature Regression Pipeline with Linear, Poly, Ridge Models")

features = ["Adult mortality", "BMI", "GDP", "Alcohol"]

models = {
    "Linear": joblib.load("Linear_model.pkl"),
    "Poly": joblib.load("Poly_model.pkl"),
    "Ridge": joblib.load("Ridge_model.pkl")
}

st.sidebar.header("Input Features")

adult_mortality = st.sidebar.slider("Adult mortality", 1.0, 700.0, 150.0)
bmi = st.sidebar.slider("BMI", 1.0, 80.0, 25.0)
gdp = st.sidebar.slider("GDP", 1.0, 120000.0, 5000.0)
alcohol = st.sidebar.slider("Alcohol", 0.0, 20.0, 5.0)

input_data = pd.DataFrame(
    [[adult_mortality, bmi, gdp, alcohol]],
    columns=features
)

selected_model = st.selectbox("Select Model", ["Linear", "Poly", "Ridge"])

prediction = models[selected_model].predict(input_data)[0]

st.subheader("Predicted Life Expectancy")
st.markdown(
    f"<h1 style='text-align:center;'>{prediction:.2f} years</h1>",
    unsafe_allow_html=True
)

st.subheader("Model Performance Comparison")

performance = pd.read_csv("model_performance.csv")
st.dataframe(performance)

st.subheader("Test R2 Score Bar Chart")

fig, ax = plt.subplots()
ax.bar(performance["Model"], performance["Test R2"])
ax.set_xlabel("Model")
ax.set_ylabel("Test R2 Score")
ax.set_title("Test R2 Comparison")

st.pyplot(fig)
