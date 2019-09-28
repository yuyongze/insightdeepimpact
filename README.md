# Deep Impact

Improve Movie Discourse, Better!

 A web application to help you evaluate and improve the impact of movie reviews. 

Links: [https://insightdeepimpact.com](https://insightdeepimpact.com)

## Project Description

This project was built by Yongze Yu at Insight Data Science during the Autumn 2019 Boston session. 

Review experiences have been deemed as very important features for ecommerce business. That is especially true on movies business, because the movie reviews make a great impact for people to decide whether they are willing to watch it. This project is aimed to forecast the impact/helpfulness of the review based on the review text before the users posting it . 

## Data and Tools

### Data Collection

The data is collected and scraped from  [IMDB](https://www.imdb.com/) using [Selenium](https://www.seleniumhq.org/) web driver and [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/). 100,000+ reviews has been collected on 1000 movies released on [2018](https://www.imdb.com/search/title/?title_type=feature&year=2018-01-01,2018-12-31). 

### Data Preprocessing

The review text is preprocessed to general structural, syntax, topic, lexical, and content information using [Scikit-Learn Text Preprocessing](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html), [Natural Language Toolkit](https://www.nltk.org/), and [Gensim](https://radimrehurek.com/gensim/index.html).

