# Sentiment Analysis on Video Game Reviews

# Research and Selection of Methods
### Objectives
Our primary objective is to design and implement a robust pipeline that processes raw textual data, extracts meaningful features, and employs state-of-the-art classification algorithms to gauge sentiment accurately. This approach aims to provide actionable insights that can influence game development, marketing strategies, and overall industry decision-making.

The central challenge lies in interpreting the varied and often nuanced language found in game reviews. Reviews may include slang, domain-specific jargon, and subtle emotional cues that complicate sentiment classification. Overcoming these hurdles involves rigorous data cleaning, text normalization, and feature extraction. Our solution will harness both traditional machine learning and modern deep learning techniques to achieve high accuracy in sentiment analysis.

Dataset overview:

|   id |      title |                                quote |                                             score |  date |   platform |      author | publicationName | review_type |      |
| ---: | ---------: | -----------------------------------: | ------------------------------------------------: | ----: | ---------: | ----------: | --------------: | ----------: | ---- |
|    2 | 1300001290 | The Legend of Zelda: Ocarina of Time | an open world on the classic console, perhaps ... | 100.0 | 2023-02-03 | Nintendo 64 |      aaronnmp96 |         NaN | user |
|    3 | 1300001290 | The Legend of Zelda: Ocarina of Time | This masterpiece holds a special place in the ... | 100.0 | 2022-10-24 | Nintendo 64 | AmadouIraklidis |         NaN | user |
|    4 | 1300001290 | The Legend of Zelda: Ocarina of Time | 10 out of 10 easily one of the best games of h... | 100.0 | 2023-01-31 | Nintendo 64 |          slushy |         NaN | user |
|    5 | 1300001290 | The Legend of Zelda: Ocarina of Time | Absolutely the best game ever made. Completely... | 100.0 | 2022-06-12 | Nintendo 64 |      Konnor1224 |         NaN | user |
|    6 | 1300001290 | The Legend of Zelda: Ocarina of Time | the people rating this game a 0 will absolutel... | 100.0 | 2022-11-21 | Nintendo 64 |   Pokemandeluxe |             |      |


### Literature Review
### Benchmarking

Multiple models were tested to determine the models to move forward with for development. The time and accuracy of each model was explored to determine the best model to start developing. 

### Preliminary Experiments
Preliminary experiments were done on a subset of the data to get an idea of the time and resources required. The distilbert for sequence classification was used to gain an understanding of performance and resource requirements (which turned out to be quite extensive). The same model that was used for sampling, crashed the computer when training on the full dataset and had to be adjusted.

# Model Implementation
### Framework Selection
There were multiple frameworks that were used in the construction of the model:

##### Tensorflow

- Primary framework to structure the data and build the model

- `tf.data.Dataset` to store the data in a dataset for processing the data through the model
- `tf.keras.layers` to add additional layers to the base model to prevent overfitting and better train the model

##### HuggingFace

- `DistilBertTokenize` - used to tokenize the data into an appropriate format for the distilbert model.

- `TFDistilBertModel`  - this model was used as the inital layer that the data was trained on to be then passed into additional layers that were implemented.
### Dataset Preparation
#### Logistic Regression

The data preparation for logistic regression inlcudes multiple steps:

- lowercase text
- remove punctuation
- tokenize words
- removing stop words
- lemmatizing words


The dataset chosen includes video game reviews from a myriad of languages. The focus of this project was to just look at the ‘English’ language reviews. The dataset first had to be filtered using a python library `langdetect` in order to label the language of each review. Additionally, each score was on a scale from 1-100, in order to perform sentiment analysis these scores were bucketed into values for 'negative', 'neutral', and 'positive' to represent sentiment categories.

#### Distilbert Transformer
Transfomers are capable of being able to capture the context and dependencies in words, and perform well with minimal data cleaning. It is generally not required to do extensive data cleaning on transformers so minimal cleaning was done:

- Removed leading, trailing, and extra whitespaces

#### Tokenization:

- `Distilbert-base-uncased` tokenizer was loaded and implemented to tokenize the text data for model training
- Function `tokenize_optimized` to tokenize data in batches to avoid memory crashing on the computer.
  - `MAX_LEN` set to 128
  - `truncation` = True to truncate longer strings of texts
  - `padding` = 'max_length' to add padding to the end of text strings that are shorter than the set max_length
  - `attention_masks` = True, to distinguish between padded and real tokens
  - `special_tokens`= True, to add tokens [CLS, SEP] to distinguish each sequence for classification
- Function outputs the input_ids and attention_masks for each row of the quote column in the dataframe

#### Data Splitting

- Data was split using `train-test-split` to first split data into training and test data and then the training data was further split to get validation data.

#### Sentiment Encoding:

- Sentiment was split into 'positive', 'negative', and 'neutral'. These values were then encoded into numerical values for training the model:

  ```
  sentiment_mapping = {
      'negative': 0,
      'neutral': 1,
      'positive': 2
  }
  ```



#### Dataset creation:

The model was created using a distilbert model with additional layers added on to try and avoid overfitting and improve performance of the model. The additional layers that were added need the data to be in a dataset format with input_ids and attention_masks that can be passed into the distilbert layer and then output to the next layer:

```
train_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': input_ids, 'attention_mask': attention_mask},
    train_labels
))
train_dataset = train_dataset.shuffle(100).batch(16)
```

```
val_dataset = tf.data.Dataset.from_tensor_slices((
    {'input_ids': val_input_ids, 'attention_mask': val_attention_mask},
    val_labels
))
val_dataset = val_dataset.batch(16)
```


### Model Development

The model was created using a distilbert model with additional layers added on to try and avoid overfitting and improve performance of the model. The additional layers that were added need the data to be in a dataset format with input_ids and attention_masks that can be passed into the distilbert layer and then the hidden states can be output to the next layer.

Model layout:

- input layer with `attention_masks` and `input_ids`
- bert layer:  ouputs the `hidden_state` from the pretrained bert model to pass to the next layer

- dropout layer: regulariztion to avoid overfitting
- global average pooling layer: to reduce the dimensions of the data to produce an output that can be passed to the dense layer
- dense layer: final dense output layer that takes the input from the pooling layer and uses softmax to then predict probabilities
### Training and fine-tuning
Due to the computational requirements and tendency to overfit the training callbacks were implemented to look at the validation accuracy and stop the model training if the validation accuracy didn't improve after 2 epochs:

```
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=2,
        restore_best_weights=True)]
```



### Evaluation and Metrics
