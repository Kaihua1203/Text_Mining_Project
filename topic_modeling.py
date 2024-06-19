from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from preprocessing import run_preprocessing
from result_visualization import run_plot_topwords

def run_topic_modeling(text):
    # Preprocessing and Wordembedding(bags of words)
    text_preprocessed = text.apply(run_preprocessing)
    vectorizer = CountVectorizer(max_df=0.90, min_df=2, stop_words="english")
    text_preprocessed = vectorizer.fit_transform(text_preprocessed)

    # Apply LDA topic modeling
    num_topics = 6  # Specify the number of topics you want to discover
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_features = lda_model.fit_transform(text_preprocessed)

    # Print topic top words
    feature_names = vectorizer.get_feature_names_out()
    for topic_idx, topic in enumerate(lda_model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-15:-1]]))

    # plot the top words
    n_topword = 15
    title = "Topics in LDA model"
    path = "./images/Topic Modeling.png"
    run_plot_topwords(lda_model, n_topword, feature_names, title, path)



    