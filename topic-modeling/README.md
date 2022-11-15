### TOPIC MODELING

Topic modeling is a statistical modeling that can be used to discover hidden (latent) themes from the collection of documents. Having information about the problems and opinions about a certain product or services can be very important for businesses. But discovering themes in a large mine of online reviews or comments in the internet can be a expensive process. Thus, several algorithms like NMF, LDA, BERT can be employed for this purpose. In this project, I'll be discussing about LDA.

LDA extracts latent (hidden) topics from the corpus. It assumes that the corpus is the mix of distribution of topics and topics is the mix of distribution of words. 

Dataset: 

The dataset scraped from reddit under title 'Depression' was used for this purpose. The link to scrape reddit post is available [here](https://github.com/shikshya1/projects/tree/main/reddit-scraper)

Data Preprocessing:

1) Split the data into sentences and sentences into word. Lowercase the words and remove the words with length less than two characters. 
2) Remove stopwords by looking at the default set given by NLTK. The stopword's list was extended by looking at the words distribution and removing the ones that didnot add much meaning to the analysis.
3) The digits were also removed from the dataset.
4) Words were lemmatized before passing on to the LDA model.
5) Bigrams are created.

Text representation:

Dictionary and the corpus are the input passed to the LDA model. Created a dictionary from the preprocessed text. Filter out the tokens that appear in less than 5 documents and more than 90 documents.Use gensim doc2bow function to create dictionary that contains information about how many times the word appears.

Building LDA:

Params:
1) Number of topic
2) alpha : document-topic density 
3) eta: topic-word density

Interpreting topics:
{{< figure src="depression_topics.png" >}}

1) Topic 1: Topic 1 mainly has words like disorder, death,shame, lonelineness, illness which can be categorized as loss.
2) Topic 2: Topic 2 mainly has discussion around mental health topic with words like depression, anxiety, trauma, help, adhd and so on.
3) Topic 3: Topic 3 talks about relationship with words like parent, mom, brother, sister, sibling, best friend , daughter, cousin and so on
4) Topic 4: The words in topic 4 can be categorized as daily struggles as it contains words like school,course, college, work, career, weight  and so on
5) Topic 5: Topic 5 contains the most miscellaneous words out of other topics. Words under several themes like world, mind, life, family, friend, feel, reason and so on. It contains themes from mental health to realtionship.

BERTopic: BERTopic uses contextual embeddings that can capture the contextual nature of the text. The structure of BERTopic (embeddings, UMAP, HBDSCAN, c-TF-IDF) can be used accordingly to adapt to the current advancements being made in language models, clustering algorithms and dimensionality reduction algorithm.


BERTopic vs LDA:

In cases where computing sentence embeddings can be expensive, LDA is preferred over BERTopic. The interpretation of topic plays a very major role in topic modeling. So, the evaluation of which technique is better is subjective based on the use cases.  Both techniques can be employed to create useful topic represntations.

The code for the implementation of LDA and BERTopic is linked [here]

