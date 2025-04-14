from bertopic import BERTopic
import gensim.corpora as corpora
from pprint import pprint
import nltk
import pandas as pd
from nltk.corpus import stopwords
import spacy
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from matplotlib import pyplot as plt
from sentence_transformers import SentenceTransformer
from umap import UMAP
from bertopic.vectorizers import ClassTfidfTransformer


# download the nltk packages
nltk.download('punkt')
nltk.download('stopwords')

# Input the English data #
df_English = pd.read_csv("D:\PyCharm 2023.1\\new_reddit\datasets\\reddit_posts.csv")

# delate the NA posts and combine the body column with the title column
df_English.dropna(subset=['body'], inplace=True)
df_English["Combined"] = df_English.apply(lambda row: f"{row['Title']} - {row['Body']}", axis=1)

df_English['Combined'] = df_English['Combined'].astype(str)   # transfer 'Abstract' into strings

# check the dataset
print(df_English.head(5))


# Remove stopwords #
stop_words = set(stopwords.words('english'))

# define a function to remove stopwords
def remove_stopwords(abstract):
    # convert all strings into lowercases
    abstract_lower = abstract.lower()

    # tokenization
    words = nltk.word_tokenize(abstract_lower)

    # remove stopwords
    text_removed = [word for word in words if word not in stop_words]

    # return the processed texts
    return text_removed

processed_abstract_list = []
for each_abstract in df_English['Abstract']:
  processed_abstract = remove_stopwords(each_abstract)
  processed_abstract_list.append(processed_abstract)

processed_stringformat = [' '.join(map(str, lst)) for lst in processed_abstract_list]


# Load the English model
nlp = spacy.load('en_core_web_sm')
# a new list for lemmatized sentences
lemmatized_sentence = []
for each_sentence in processed_stringformat:
    lemmatized_words_list = []
    each_doc = nlp(each_sentence)
    for token in each_doc:
        if token.pos_ in ('NOUN', 'ADV', 'PROPN', 'VERB', 'ADJ'):
            lemmatized_words_list.append(token.lemma_)
    lemmatized_sentence.append(' '.join(lemmatized_words_list))

print("Now we have {} rows of data".format(len(lemmatized_sentence)))


# Extract embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Reduce dimensionality
umap_model = UMAP(n_neighbors=100, n_components=10, min_dist=0.0, metric='cosine')

# Remove stopwords and set the N-gram range
vectorizer_model = CountVectorizer(ngram_range=(1, 4), stop_words="english")

# Create topic representation
ctfidf_model = ClassTfidfTransformer()

# Set the hyperparameters of topic model
topic_model = BERTopic(
    language='English',
    min_topic_size=10,
    embedding_model=embedding_model,
    umap_model=umap_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model
    )

# calculate the perplexity
perplexity_values = []
for iteration in range(20):
    topics, probs = topic_model.fit_transform(lemmatized_sentence)
    log_perplexity = -1 * np.mean(np.log(np.sum(probs, axis=1)))
    perplexity = np.exp(log_perplexity)
    perplexity_values.append(perplexity)

# 绘制 perplexity 曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, 20 + 1), perplexity_values, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Perplexity')
plt.title('Perplexity Curve')
plt.grid(True)
plt.show()


# Train the model
topics, probs = topic_model.fit_transform(lemmatized_sentence)


# check topic information
freq = topic_model.get_topic_info()
print("--------------------topic info--------------------")
print(freq)

# the default labels for each topic
print(topic_model.topic_labels_)

# the size of each topic
print("the size of each topic: ", topic_model.topic_sizes_)

# visualize Topics
print(topic_model.visualize_topics())
plt.savefig("D:\PyCharm 2023.1\\new_reddit\\topic_visualization.png")

# visualize topic hierarchy
topic_model.visualize_hierarchy(top_n_topics=10)
plt.savefig("D:\PyCharm 2023.1\\new_reddit\\topic_hierarchy.png")

# visualize the topics via barchart
topic_model.visualize_barchart(top_n_topics=8)
plt.savefig("D:\PyCharm 2023.1\\new_reddit\\topic_barchart.png")

# check keywords of every topic
for i in range(9):
    print(i)
    print(topic_model.get_topic(i))

