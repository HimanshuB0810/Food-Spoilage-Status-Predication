
# Food Spoilage Status Prediction

This project is a machine learning application that predicts the spoilage status of food based on environmental factors. It uses a K-Nearest Neighbors (KNN) model and a Streamlit web interface for user interaction.

---

## Features

* **User-Friendly Interface**: A simple web interface built with Streamlit allows users to input data and get predictions easily.
* **Machine Learning Model**: The prediction is powered by a K-Nearest Neighbors (KNN) classification model trained on a food spoilage dataset.
* **Real-time Predictions**: The application provides real-time spoilage status predictions based on the input values.

---

## Installation

To run this project, you need to have Python installed. You can then install the required libraries using pip:

```bash
pip install -r requirements.txt
````

The necessary libraries are:

  * streamlit
  * pandas
  * numpy
  * seaborn
  * matplotlib
  * scikit-learn

-----

## Usage

To start the application, navigate to the project directory in your terminal and run the following command:

```bash
streamlit run app.py
```

The web application will open in your browser. You can then input the following values to get a spoilage status prediction:

  * **Ethylene (ppm)**
  * **CO2 (ppm)**
  * **Temperature (C)**
  * **Humidity (%RH)**

The application will output a "Spoil Status" of either 0 (not spoiled) or 1 (spoiled).

-----

## Project Structure

  * **app.py**: The main application file that runs the Streamlit web interface.
  * **experiment.ipynb**: A Jupyter Notebook containing the experimental code, data analysis, and model training process.
  * **food\_spoilage\_dataset .xlsx - Sheet1.csv**: The dataset used to train the model.
  * **model.pkl**: The saved K-Nearest Neighbors model.
  * **scaler.pkl**: The saved StandardScaler object for scaling the input data.
  * **dataframe.pkl**: The saved training data.
  * **requirements.txt**: A list of the required Python libraries for this project.
  * **.gitignore**: A file specifying which files and folders to ignore in a Git repository.

-----

## Model

The project uses a **K-Nearest Neighbors (KNN)** classification model to predict food spoilage. The model was chosen after evaluating several other classification algorithms, as shown in the `experiment.ipynb` notebook. The KNN model achieved an accuracy of 97.5% on the test set.

-----

## Dataset

The model was trained on the `food_spoilage_dataset .xlsx - Sheet1.csv` dataset. The dataset contains the following features:

  * **Ethylene (ppm)**
  * **CO2 (ppm)**
  * **Temperature (C)**
  * **Humidity (%RH)**
  * **Spoilage\_Status** (Target Variable)

The dataset is balanced, with an equal number of spoiled and not spoiled samples.

```