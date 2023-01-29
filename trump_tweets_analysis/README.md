## Extracting entities mentioned in the tweets posted by Former U.S president Trump, analyze the sentiment associated with it and visualize the data using Tableau.

Millions of individuals, including politicians, use the well-liked social networking site Twitter to tweet about their ideas and experiences. Because of this, there is a substantial amount of text data available in the form of tweets that may be examined and used to learn important lessons. Extracting entities from the tweets, such as names of people, organizations, locations, and more, is one method of analyzing this text data. An additional method is to conduct sentiment analysis, which is the process of figuring out the sentiment or emotion communicated in a text.

Using Flair, a Python natural language processing module, we will explore how to extract entities from tweets by former US President Donald Trump and analyze sentiment associated with the extracted entity. 

The objectives of this posts are:

1) Extract organizations mentioned in the tweets
2) Assign sentiment associated with the extracted entities
3) Visualize the tweets using Tableau

#### Data Exploration:

Data source: http://trumptwitterarchive.com/

The twitter account of Donald Trump was deactivated on Jan 8,2021 to prevent the glorification of violence. But there are some tweets registered after that date are present in the dataset. So, the first step will be to examine the data. For more detail, please refer to the code. 

After cleaning the date, the latest date of tweet is Jan 8,2021. The mentions of twitter handles and URLs are also removed.

#### Extracting entities from the tweets

After loading the entity recognition model from Flair, a sentence is created from the text of the tweet, and then the entity recognition is applied to the sentence. The result displays each entity in the sentence along with its type and position within the text.

```
def get_orgs(text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    org_list = [entity.text for entity in sentence.get_spans('ner') if entity.tag == 'ORG']

    org_list = list(set(org_list))
    return org_list

```

Only organizations are extracted from the tweets. We will use Counter to create a frequency table of organization mentions. The most common 10 organizations mentioned in the tweets are:

| Words            | Frequency     |
| ----------------:|:-------------:| 
| Congress         | 376           | 
| Fake News        | 311           |  
| Senate           | 279           | 
| FBI              | 257           | 
| Republican Party | 195           | 
| House            | 189           | 
| Fake News Media  | 167           | 
| ISIS             | 151           | 
| CNN              | 149           | 
| Trump Tower      | 133           | 

![alt text][freq-words]

[freq-words]: https://github.com/shikshya1/projects/blob/main/trump_tweets_analysis/images/frequent_words.png?raw=true


#### Assigning sentiment to the tweets

Text classifier model of flair is initialized to perform sentiment analysis. The output shows the sentiment score of the sentence along with the sentiment label. 

```
def get_sentiment(text):
    sentence = flair.data.Sentence(text)
    model.predict(sentence)
    sentiment = sentence.labels[0]
    return sentiment

```

We will then calculate average positive and positive score for each entity identified in the tweets.

The negative sentiments in the tweets were mostly associated with Democratic party, T-Mobile, Networks that trump classified as Clinton News Network	providing fake news like New york times, washington post. NBA also has been associated with negative tweets.

The most positive entities were associated with Baseball Hall of Fame, Lincoln Memorial, Mar-a-Lago Club, Trump Int'l Hotel  etc. Along with that Fox Networks had positive sentiment in trump's tweet. 

The CSV with the sentiment associated with the entities can be further analyzed to gain better understanding of trump's tweet.

#### Visualization using Tableau

