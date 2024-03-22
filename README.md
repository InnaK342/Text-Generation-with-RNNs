# Text Generation with RNNs

This repository contains code for text generation using Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM) units.

## Files
- **Text_Generation_with_RNNs.ipynb**: Jupyter Notebook containing the code for text generation with LSTM-based RNNs.
- **text.txt**: Text file containing the input text data used for training the RNN model.
- **wordLSTM.pth**: PyTorch model file containing the trained weights of the LSTM-based RNN model.
- **word_to_index.p**: Pickle file containing the mapping of words to their corresponding indices used in the model.

## Project Description
This project demonstrates text generation using LSTM-based RNNs. The goal is to train an RNN model to learn the structure of a given input text and generate new text that resembles the original text. This technique finds applications in natural language processing (NLP) tasks such as language modeling, text summarization, and dialogue generation.

## Code Overview
The main components of the code include:
1. Data Preprocessing: Loading the input text data, tokenizing, and cleaning it.
2. Dataset Preparation: Generating training data by converting text into sequences of indices.
3. Model Definition: Defining the architecture of the LSTM-based RNN model.
4. Model Training: Training the RNN model using the prepared dataset.
5. Text Generation: Using the trained model to generate new text.

## Model Training
The RNN model is trained for a specified number of epochs using the Adam optimizer with cross-entropy loss. The training progress and average loss are printed after each epoch.

## Generating Text
After training, the model can be used to generate new text by providing a seed sequence as input. The model predicts the next words iteratively based on the input sequence, generating text that resembles the original data.

For more details, refer to the Jupyter Notebook `Text_Generation_with_RNNs.ipynb`.
