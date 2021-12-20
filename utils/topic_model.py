from top2vec import Top2Vec
import re
from bs4 import BeautifulSoup
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from wordcloud import WordCloud

nltk.download('wordnet')

global wnl
wnl = WordNetLemmatizer()

# list of custom stopwords
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't", 'hi', 'okay', 'ok', 'ohkay', 'bro', 'bye', 'thanks', 'thank', 'yeah', 'ya', \
            'u', 'ur', ])


# https://stackoverflow.com/a/47091490/4084039
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def preprocess_text(sentence:str):
    #a. remove html and url tags from text
    sentence = re.sub(r"http\S+", "", sentence)
    sentence = BeautifulSoup(sentence, 'lxml').get_text()

    #b.expand contracted terms
    sentence = decontracted(sentence)

    #c.remove non aplhabet characters
    sentence = re.sub("\S*\d\S*", "", sentence).strip()
    sentence = re.sub('[^A-Za-z]+', ' ', sentence)

    #d. lemmatize each word in sentence
    #e. and turn them into lower case
    #list of stop words: https://gist.github.com/sebleier/554280
    sentence = ' '.join(wnl.lemmatize(word.lower()) for word in sentence.
    split() if word.lower() not in stopwords)

    return sentence

def get_topics(df):
    """
    Preprocesses conversations to prepare it for Topic modelling
    Returns list of wordclouds of top two topics
    """

    df = df[df.message != '<Media omitted>']
    df = df[df.message != 'This message was deleted']

    documents = df['message'].apply(preprocess_text)
    model = Top2Vec(documents=documents.tolist(), speed="learn", workers=-1)

    num_topics = model.get_num_topics()
    if num_topics >= 2:
        topic_words, word_scores, topic_nums = model.get_topics(2)

    elif num_topics == 1:
        topic_words, word_scores, topic_nums = model.get_topics(1)
    
    else:
        return []

    clouds = []
    for topic in topic_words[:2]:
        wc = WordCloud(width=700, height=300, min_font_size=12, background_color='white')
        wc = wc.generate(' '.join(topic))
        clouds.append(wc)

    del model
    del documents
    return clouds
