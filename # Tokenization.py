# Tokenization
tokens = word_tokenize(text)

# Lowercase
tokens = [word.lower() for word in tokens]

# Remove punctuation
tokens = [word for word in tokens if word not in string.punctuation]

# Remove stopwords
stop_words = set(stopwords.words("english"))
tokens = [word for word in tokens if word not in stop_words]

# Stemming
stemmer = PorterStemmer()
tokens = [stemmer.stem(word) for word in tokens]

# Lemmatization
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens]

# Join tokens back into a string
clean_text = " ".join(tokens)

# Bag of Words
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(preprocessed_tweets)

# TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(preprocessed_tweets)

# Bigrams
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))
X_bigram = bigram_vectorizer.fit_transform(corpus)

# Combine two tweet lists
all_tweets = positive_tweets + negative_tweets

# Create labels
labels = [1] * len(positive_tweets) + [0] * len(negative_tweets)

# Apply preprocessing to all tweets
preprocessed_tweets = [" ".join(preprocess_text(tweet)) for tweet in all_tweets]
