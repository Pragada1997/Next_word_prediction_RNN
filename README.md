# Shakespeare Hamlet Next Word Prediction Using LSTM

## 📌 Project Overview

This project implements a **Next Word Prediction system** using **Natural Language Processing (NLP)** and **Deep Learning**.

The model is trained on **William Shakespeare's *Hamlet*** text obtained from the **NLTK Gutenberg corpus**. The project demonstrates how raw text can be processed, tokenized, converted into N-Gram sequences, and used to train a recurrent neural network for predicting the next word in a sentence.

The current implementation uses a **Long Short-Term Memory (LSTM)** neural network with an **Embedding layer** and a **Softmax output layer**.

> **Note:** The current notebook implements the LSTM version. GRU, Dropout, EarlyStopping, independent test evaluation, and LSTM-vs-GRU comparison are identified as future improvements because they are not implemented in the current notebook.

---

## 🎯 Project Objectives

The main objectives of this project are:

* Load Shakespeare's *Hamlet* using the NLTK Gutenberg dataset.
* Clean and preprocess the text.
* Tokenize the text and create a word index.
* Convert words into numerical representations.
* Create N-Gram sequences.
* Prepare input and target sequences.
* Apply sequence padding where required.
* Build an LSTM-based language model.
* Use an Embedding layer to represent words as dense vectors.
* Train the model using categorical crossentropy and Adam optimizer.
* Save the best-performing model using ModelCheckpoint.
* Create a custom Next Word Prediction function.
* Test the model using user-provided text.

---

## 📚 Dataset

The dataset used in this project is Shakespeare's *Hamlet*, available through the **NLTK Gutenberg corpus**.

The dataset does not need to be downloaded manually.

### Dataset Loading

```python
import nltk

nltk.download('gutenberg')

from nltk.corpus import gutenberg

raw_data = gutenberg.raw('shakespeare-hamlet.txt')
```

---

## 🔄 Project Workflow

The overall project workflow is:

```text
Hamlet Text
     ↓
Text Preprocessing
     ↓
Tokenization
     ↓
Word Index / Vocabulary
     ↓
Integer Encoding
     ↓
N-Gram Sequence Creation
     ↓
Input (X) and Target (y)
     ↓
One-Hot Encoding
     ↓
LSTM Model
     ↓
Model Training
     ↓
Model Checkpointing
     ↓
Load Best Model
     ↓
User Input
     ↓
Next Word Prediction
```

---

# 🧹 1. Data Preprocessing

The raw Shakespeare text contains newline characters, carriage returns, BOM characters and apostrophes.

The notebook performs basic cleaning using:

```python
data = raw_data.replace('\n', '') \
               .replace('\r', '') \
               .replace('\ufeff', '') \
               .replace("'", '')
```

This produces a cleaner text corpus for tokenization.

---

# 🔤 2. Tokenization

The cleaned text is tokenized using Keras `Tokenizer`.

Tokenization converts words into numerical IDs that can be processed by a neural network.

For example:

```text
"I think I hear"
```

may be represented as:

```text
[1, 637, 4, 45]
```

The tokenizer also creates a **word index**, mapping each word to an integer.

### Dataset Statistics

| Metric                |  Value |
| --------------------- | -----: |
| Token sequence length | 28,764 |
| Vocabulary size       |  5,449 |
| Generated sequences   | 28,761 |

---

# 🔢 3. N-Gram Sequence Generation

The project uses N-Gram sequences to convert the text into supervised learning examples.

Four consecutive tokens are used to create each sequence.

Conceptually:

```text
[word1, word2, word3, word4]
```

is converted into:

```text
Input X:
[word1, word2, word3]

Target y:
[word4]
```

Therefore, the model learns:

```text
Previous 3 words → Next word
```

For example:

```text
"I am going"
```

could be used to predict:

```text
"home"
```

---

# 📐 4. Input and Target Preparation

The generated sequences are separated into:

```python
X = sequences[:, :-1]
y = sequences[:, -1]
```

The target word is then converted into a one-hot encoded vector using:

```python
from tensorflow.keras.utils import to_categorical

y = to_categorical(y, num_classes=total_words)
```

This transforms the problem into a **multi-class classification task**, where every vocabulary word represents a possible output class.

---

# 🧠 5. LSTM Model Architecture

The project uses a stacked LSTM architecture.

### Model Architecture

```text
Input Sequence
     ↓
Embedding Layer
     ↓
LSTM Layer – 1000 units
     ↓
LSTM Layer – 1000 units
     ↓
Dense Layer – 1000 units
     ↓
ReLU Activation
     ↓
Dense Layer – 5449 units
     ↓
Softmax
     ↓
Predicted Next Word
```

### Architecture Details

| Layer             | Configuration                      |
| ----------------- | ---------------------------------- |
| Embedding         | 5,449 vocabulary × 10 dimensions   |
| LSTM              | 1,000 units, return_sequences=True |
| LSTM              | 1,000 units                        |
| Dense             | 1,000 units                        |
| Activation        | ReLU                               |
| Output Dense      | 5,449 units                        |
| Output Activation | Softmax                            |

The model contains approximately:

**18.56 million trainable parameters.**

---

# 🔡 6. Embedding Layer

The Embedding layer converts integer word IDs into dense numerical vectors.

Instead of representing a word using a large one-hot vector, the model represents each word using a compact vector of size 10.

```text
Word ID
   ↓
Embedding
   ↓
10-dimensional vector
```

The embedding representation can help the neural network learn relationships and patterns between words.

---

# 🔁 7. LSTM

LSTM stands for **Long Short-Term Memory**.

LSTM is a type of Recurrent Neural Network designed to learn dependencies in sequential data.

For language modeling, the sequence of words is important because earlier words can provide context for predicting the next word.

The LSTM architecture uses memory mechanisms to retain useful information while processing the sequence.

---

# ⚙️ 8. Model Compilation

The model is compiled using:

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001)
)
```

### Loss Function

`categorical_crossentropy` is used because the target is represented as a one-hot encoded vector across the vocabulary.

### Optimizer

The **Adam optimizer** is used to update the neural-network weights during training.

The learning rate is:

```text
0.001
```

---

# 🏋️ 9. Model Training

The model is trained using:

```python
model.fit(
    X,
    y,
    epochs=2,
    batch_size=64,
    callbacks=[checkpoint]
)
```

### Training Configuration

| Parameter     |                    Value |
| ------------- | -----------------------: |
| Epochs        |                        2 |
| Batch Size    |                       64 |
| Optimizer     |                     Adam |
| Learning Rate |                    0.001 |
| Loss          | Categorical Crossentropy |

The notebook reports **450 batches per epoch**.

---

# 💾 10. Model Checkpointing

`ModelCheckpoint` is used to save the best model during training.

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    "next_words.h5",
    monitor="loss",
    verbose=1,
    save_best_only=True
)
```

The model monitors the training loss and saves the model whenever the loss improves.

### Training Results

| Epoch   | Training Loss |
| ------- | ------------: |
| Epoch 1 |       6.93918 |
| Epoch 2 |       6.53442 |

The reduction in loss indicates that the model learned patterns from the training corpus during the two training epochs.

---

# 🔮 11. Next Word Prediction

After training, the saved model and tokenizer can be used to predict the next word.

The prediction process is:

```text
User Input
    ↓
Tokenization
    ↓
Convert words → Token IDs
    ↓
Keep required context
    ↓
LSTM Model
    ↓
Probability Distribution
    ↓
Argmax
    ↓
Predicted Word
```

The model produces probabilities for all words in the vocabulary.

The word with the highest probability is selected as the predicted next word.

Conceptually:

```python
prediction = model.predict(sequence)

predicted_index = np.argmax(prediction)
```

The predicted index is then converted back into the corresponding word using the tokenizer's word index.

---

# 🧪 12. Example Prediction

The notebook demonstrates a prediction using:

```text
I thinke I heare
```

The model predicts:

```text
the
```

Therefore:

```text
Input:
"I thinke I heare"

Predicted next word:
"the"
```

---

# 📊 13. Results

The current implementation successfully demonstrates the complete basic pipeline for LSTM-based next-word prediction.

### Key Results

```text
Vocabulary Size       : 5,449
Token Sequence Length : 28,764
Training Sequences    : 28,761
Epochs                : 2
Batch Size             : 64

Epoch 1 Loss           : 6.93918
Epoch 2 Loss           : 6.53442
```

### Prediction Example

```text
Input:
I thinke I heare

Prediction:
the
```

---

# ⚠️ 14. Current Implementation Limitations

The project requirements include several techniques that are not present in the current notebook implementation.

| Requirement                  | Current Status         |
| ---------------------------- | ---------------------- |
| Hamlet Dataset               | ✅ Implemented          |
| Text Preprocessing           | ✅ Implemented          |
| Tokenization                 | ✅ Implemented          |
| Word Index                   | ✅ Implemented          |
| N-Gram Sequences             | ✅ Implemented          |
| Sequence Padding             | ⚠️ Not explicitly used |
| LSTM                         | ✅ Implemented          |
| GRU                          | ❌ Not implemented      |
| Embedding                    | ✅ Implemented          |
| Dropout                      | ❌ Not implemented      |
| Early Stopping               | ❌ Not implemented      |
| Train/Test Split             | ❌ Not implemented      |
| Independent Model Evaluation | ⚠️ Limited             |
| Next Word Prediction         | ✅ Implemented          |
| LSTM vs GRU Comparison       | ❌ Not implemented      |

The current notebook uses fixed three-word input sequences, so explicit padding is not required for those inputs.

Also, because an independent validation/test dataset is not created, the reported loss represents **training performance**, rather than performance on unseen data.

---

# 🚀 15. Future Improvements

The following improvements can make the project more complete and robust:

### 1. GRU Model

Build a GRU model with an architecture comparable to the LSTM model.

```text
Embedding
    ↓
GRU
    ↓
GRU
    ↓
Dense
    ↓
Softmax
```

### 2. Dropout

Add Dropout layers to help reduce overfitting.

```python
Dropout(0.2)
```

### 3. Early Stopping

Use validation loss with EarlyStopping:

```python
EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

### 4. Train/Test Split

Separate the dataset into training and testing data so that the model can be evaluated on unseen sequences.

### 5. LSTM vs GRU Comparison

Train both models using the same data and compare:

* Training loss
* Validation loss
* Test loss
* Prediction accuracy
* Training time
* Number of parameters

### 6. Better Text Preprocessing

More advanced preprocessing can include:

* Sentence segmentation
* Punctuation handling
* Normalization
* Out-of-vocabulary handling
* More consistent whitespace processing

---

# 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NLTK
* NumPy
* Pandas
* Google Colab
* Recurrent Neural Networks
* LSTM
* NLP
* Word Embeddings

---

# 📁 Project Structure

A recommended GitHub repository structure is:

```text
Shakespeare-Hamlet-Next-Word-Prediction/
│
├── RNN_Next_word_predict_project.ipynb
├── README.md
├── next_words.h5
├── token.pkl
└── requirements.txt
```

> The `.h5` model file and `token.pkl` should only be committed if you intend to distribute the trained model and tokenizer with the project.

---

# ▶️ How to Run the Project

### Step 1 — Clone the repository

```bash
git clone <your-github-repository-url>
```

### Step 2 — Open the notebook

Open:

```text
RNN_Next_word_predict_project.ipynb
```

using Google Colab or Jupyter Notebook.

### Step 3 — Install dependencies

```bash
pip install tensorflow nltk numpy pandas
```

### Step 4 — Download the NLTK Gutenberg corpus

```python
import nltk

nltk.download('gutenberg')
```

### Step 5 — Run the notebook

Execute the cells sequentially from top to bottom.

---

# 📌 Conclusion

This project demonstrates how **Natural Language Processing and Recurrent Neural Networks** can be used to build a word-level language model.

Using Shakespeare's *Hamlet*, the project performs text preprocessing, tokenization, vocabulary creation, N-Gram sequence generation, one-hot encoding, and LSTM model training.

The trained model learns patterns from the Shakespeare corpus and can generate a predicted next word based on a three-word input context.

The current implementation achieves a reduction in training loss from **6.93918 to 6.53442** over two epochs and successfully demonstrates next-word prediction for a sample user input.

Future development will focus on implementing **GRU, Dropout, EarlyStopping, independent test evaluation, and a detailed LSTM-versus-GRU performance comparison**.

---

## 👨‍💻 Author

**Ramesh Pragada**

Data Science | Machine Learning | Deep Learning | NLP

---

## ⭐ Project

If you find this project useful, consider giving the repository a ⭐ on GitHub.
