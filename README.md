# Natural Language Processing Specialization

### Table of Contents
1. [Introduction](#Intro)
2. [Instructions to use the repository](#Instruction)
3. [My Learnings from the Specialization](#Learning)
4. [Programming Assignments](#Programming)
5. [Disclaimer](#Disc)

## Introduction<a name="Intro"></a>
This Specialization is designed and taught by two experts in NLP, machine learning, and deep learning. **Younes Bensouda Mourri is an Instructor of AI at Stanford University** who also helped build the Deep Learning Specialization. **Łukasz Kaiser is a Staff Research Scientist at Google Brain** and the co-author of Tensorflow, the Tensor2Tensor and Trax libraries, and the Transformer paper.

## Instructions to use the repository<a name="Instruction"></a>
Using this repository is straight forward. Clone this repository to use.
This repository contains all my work for this specialization. All the code base, quiz questions, screenshot, and images, are taken from, unless specified, [Natural Language Processing Specialization on Coursera](https://www.coursera.org/specializations/natural-language-processing).

## My Learnings from the Specialization<a name="Learning"></a>
In this four course series, I learned about **logistic regression and naïve Bayes** algorithms and implemented them from scratch in Python, used **word vectors** to implement sentiment analysis model, complete analogies (King - Man + Woman = Queen), and translate words (English to French), and used locality sensitive hashing for approximate nearest neighbors.

I used dynamic programming, hidden Markov models, and word embeddings to **autocorrect misspelled words, autocomplete partial sentences, and identify part-of-speech tags** for words.

I learned about dense and recurrent neural networks, LSTMs, GRUs, and Siamese networks in TensorFlow and Trax and used these to perform advanced **sentiment analysis, text generation, named entity recognition, and to identify duplicate questions** on Quora. 

I implemented encoder-decoder (Seq2seq model), causal, and self-attention to perform advanced **machine translation of complete sentences, text summarization, question-answering and to build chatbots**. State of the art models were taught that includes **T5, BERT, transformer, reformer**, and more!


## Programming Assignments<a name="Programming"></a>
1. **[Course 1: Natural Language Processing with Classification and Vector Spaces](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/tree/main/C1%20-%20NLP%20with%20classification%20and%20vector%20spaces)**

 - **[Week 1: Logistic Regression for Sentiment Analysis of Tweets](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C1%20-%20NLP%20with%20classification%20and%20vector%20spaces/Week_1/C1_W1_Assignment.ipynb)**
    - Classify positive or negative sentiment in tweets using Logistic Regression

 - **[Week 2: Naïve Bayes for Sentiment Analysis of Tweets](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C1%20-%20NLP%20with%20classification%20and%20vector%20spaces/Week_2/C1_W2_Assignment.ipynb)**
    - Classify positive or negative sentiment in tweets using more advanced model

 - **[Week 3: Vector Space Models](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C1%20-%20NLP%20with%20classification%20and%20vector%20spaces/Week_3/C1_W3_Assignment.ipynb)**
    - Vector space models capture semantic meaning and relationships between words. Use them to discover relationships between words, then visualize their relationships in two dimensions using PCA (dimensionality reduction technique).

 - **[Week 4: Word Embeddings and Locality Sensitive Hashing for Machine Translation](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C1%20-%20NLP%20with%20classification%20and%20vector%20spaces/Week_4/C1_W4_Assignment.ipynb)**
    - Write a simple English-to-French translation algorithm using pre-computed word embeddings and locality sensitive hashing to relate words via approximate k-nearest neighbors search.


2. **[Course 2: Natural Language Processing with Probabilistic Models](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/tree/main/C2%20-%20NLP%20with%20probabilistic%20models)**

 - **[Week 1: Auto-correct using Minimum Edit Distance](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C2%20-%20NLP%20with%20probabilistic%20models/Week_1/C2_W1_Assignment.ipynb)**
    - Create a simple auto-correct algorithm using minimum edit distance and dynamic programming

 - **[Week 2: Part-of-Speech (POS) Tagging](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C2%20-%20NLP%20with%20probabilistic%20models/Week_2/C2_W2_Assignment.ipynb)**
    - Learn about Markov chains and Hidden Markov models, then use them to create part-of-speech tags for a Wall Street Journal text corpus. Apply the Viterbi algorithm for POS tagging, which is important for computational linguistics

 - **[Week 3: N-gram Language Models](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C2%20-%20NLP%20with%20probabilistic%20models/Week_3/C2_W3_Assignment.ipynb)**
    - Learn about how N-gram language models work by calculating sequence probabilities, then build your own autocomplete language model using a text corpus from Twitter (similar models are used for translation, determining the author of a text, and speech recognition)

 - **[Week 4: Word2Vec and Stochastic Gradient Descent](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C2%20-%20NLP%20with%20probabilistic%20models/Week_4/C2_W4_Assignment.ipynb)**
    - Learn about how word embeddings carry the semantic meaning of words, which makes them much more powerful for NLP tasks, then build your own Continuous bag-of-words model that uses a neural network to compute word embeddings from Shakespeare text. 


3. **[Course 3: Natural Language Processing with Sequence Models**](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/tree/main/C3%20-%20NLP%20with%20sequence%20models)**

 - **[Week 1: Neural Networks for Sentiment Analysis](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C3%20-%20NLP%20with%20sequence%20models/Week_1/C3_W1_Assignment.ipynb)**
    - Train a neural network with GLoVe word embeddings to perform sentiment analysis on tweets

 - **[Week 2: Recurrent Neural Networks for Language Modeling](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C3%20-%20NLP%20with%20sequence%20models/Week_2/C3_W2_Assignment.ipynb)**
     - Learn about the limitations of traditional language models and see how RNNs and GRUs use sequential data for text prediction. Build your own next-word generator using a simple RNN on Shakespeare text data.

  - **[Week 3: LSTMs and Named Entity Recognition](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C3%20-%20NLP%20with%20sequence%20models/Week_3/C3_W3_Assignment.ipynb)**
    - Learn about how long short-term memory units (LSTMs) solve the vanishing gradient problem, and how Named Entity Recognition systems quickly extract important information from text. Build your own Named Entity Recognition system using an LSTM and data from Kaggle.

 - **[Week 4: Siamese Networks](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C3%20-%20NLP%20with%20sequence%20models/Week_4/C3_W4_Assignment.ipynb)**
    - Learn about Siamese networks, a special type of neural network made of two identical networks that are eventually merged together. Build your own Siamese network that identifies question duplicates in a dataset from Quora.


4. **[Course 4: Natural Language Processing with Attention Models](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/tree/main/C4%20-%20NLP%20with%20attention%20models)**

 - **[Week 1: Neural Machine Translation with Attention](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C4%20-%20NLP%20with%20attention%20models/Week_1/C4_W1_Assignment.ipynb)**
    - Discover some of the shortcomings of a traditional seq2seq model and how to solve for them by adding an attention mechanism. Build a Neural Machine Translation model with Attention that translates English sentences into German.

 - **[Week 2: Text Summarization with Transformer Models](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C4%20-%20NLP%20with%20attention%20models/Week_2/C4_W2_Assignment.ipynb)**
    - Compare RNNs and other sequential models to the more modern Transformer architecture. Build a transformer model that generates text summaries.

 - **[Week 3: Question-Answering with Transformer Models](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C4%20-%20NLP%20with%20attention%20models/Week_3/C4_W3_Assignment.ipynb)**
    - Explore transfer learning with state-of-the-art models like T5 and BERT, then build a model that can perform question answering task.

 - **[Week 4: Chatbots with a Reformer Model](https://github.com/Ankit-Kumar-Saini/Coursera_Natural_Language_Processing_Specialization/blob/main/C4%20-%20NLP%20with%20attention%20models/Week_4/C4_W4_Assignment.ipynb)**
    - Examine some unique challenges Transformer models face and their solutions, then build a chatbot using a Reformer model.

## Disclaimer<a name="Disc"></a>
The solutions uploaded in this repository are only for reference when you got stuck somewhere. Please don't use these solutions to pass programming assignments. 

