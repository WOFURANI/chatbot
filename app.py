import nltk
import string
import streamlit as st
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Make sure all needed resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
# Load NLTK resources once
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load the text file and preprocess the data
with open('./python.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

# Tokenize the text into sentences
sentences = sent_tokenize(data)


# Define a function to preprocess each sentence
def preprocess(sentence):
    # Tokenize the sentence into words
    words = word_tokenize(sentence)
    # Remove stopwords and punctuation
    words = [
        lemmatizer.lemmatize(word.lower())
        for word in words
        if word.lower() not in stop_words and word not in string.punctuation
    ]
    return words


# Preprocess each sentence in the text
corpus = [preprocess(sentence) for sentence in sentences]


# Define a function to find the most relevant sentence given a query
def get_most_relevant_sentence(query):
    # Preprocess the query
    query_tokens = preprocess(query)
    if not query_tokens:
        return "Sorry, I couldn't understand your question."

    max_similarity = 0
    most_relevant_sentence = "Sorry, I couldn't find a good answer."

    for sentence_tokens in corpus:
        if not sentence_tokens:
            continue
        similarity = len(set(query_tokens).intersection(sentence_tokens)) / float(
            len(set(query_tokens).union(sentence_tokens)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence_tokens)

    return most_relevant_sentence


# Chatbot interface function
def chatbot(question):
    return get_most_relevant_sentence(question)


# Streamlit app
def main():
    st.title("Chatbot")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the python History")

    question = st.text_input("You:")

    if st.button("Submit") and question.strip():
        response = chatbot(question)
        st.markdown(f"**Chatbot:** {response}")


if __name__ == "__main__":
    main()
