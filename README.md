# Comment Toxicity Detection

This project is used to detect toxic comments using Machine Learning and Deep Learning techniques.

The project uses the Jigsaw Toxic Comment Classification dataset. A comment can belong to more than one toxicity category, so this is a multi-label classification problem.

## Technologies Used

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Gradio
- Jupyter Notebook

## Toxicity Categories

The model predicts 6 different types of toxic comments:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

## How the Project Works

The project follows these steps:

1. Load the dataset using Pandas.
2. Separate the comments and their corresponding labels.
3. Split the data into training and validation data.
4. Convert the text into numerical sequences using TextVectorization.
5. Use an Embedding layer to represent the words.
6. Use a Bidirectional GRU to process the text.
7. Use Dense layers for classification.
8. Train the model using Binary Crossentropy.
9. Check the model using Precision, Recall and Accuracy.
10. Save the trained model.
11. Use Gradio to test comments through a simple interface.

## Model

The model used in this project contains:

- TextVectorization
- Embedding layer
- Bidirectional GRU
- Dense layers
- Sigmoid output layer

The output layer contains 6 values corresponding to the six toxicity categories.

## Dataset

The project uses the Jigsaw Toxic Comment Classification dataset.

The main dataset file used for training is:

`train.csv`

The dataset contains the comment text and labels for the different types of toxicity.

## Running the Project

First install the required libraries:

```bash
pip install tensorflow pandas numpy matplotlib scikit-learn gradio
