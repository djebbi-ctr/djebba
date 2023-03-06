import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

# Set app title
st.title("Iris Dataset Prediction")

# Load dataset
iris = datasets.load_iris()
X = iris.data
Y = iris.target

# Set classifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Add a header to the app
st.header("Enter the values below to predict the type of iris flower")

# Define input fields
sepal_length = st.slider("Sepal Length", float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider("Sepal Width", float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider("Petal Length", float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider("Petal Width", float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Set prediction button
if st.button("Predict"):
    prediction = clf.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    st.write("The predicted type of iris flower is", iris.target_names[prediction[0]])
