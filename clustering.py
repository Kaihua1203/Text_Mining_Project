from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
import pandas as pd
import matplotlib.pyplot as plt

def run_kmeans(text, wordembedding):

    if wordembedding == "1":  # Bag of Words
        vectorizer = CountVectorizer()
        data_text_vectorized = vectorizer.fit_transform(text)
        title = 't-SNE Visualization with Bag of Words'

    elif wordembedding == "2":  # TF-IDF
        vectorizer = TfidfVectorizer()
        data_text_vectorized = vectorizer.fit_transform(text)
        title = 't-SNE Visualization with TF-IDF'

    data_text_vectorized = data_text_vectorized.toarray()

    # standardization
    scaler = StandardScaler()
    text_processed = scaler.fit_transform(data_text_vectorized)

    # Building Kmeans
    kmeans = KMeans(n_clusters=6, random_state=42)
    kmeans.fit(text_processed)

    labels = kmeans.labels_
    # centers = kmeans.cluster_centers_

    # PCA reduction
    pca1 = PCA(n_components=2)
    data_pca = pca1.fit_transform(text_processed)
    data_pca_df = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
    data_pca_df['labels'] = labels

    # Visualize it
    plt.figure(figsize=(8, 6))
    plt.scatter(data_pca_df['PC1'], data_pca_df['PC2'], c=data_pca_df['labels'])
    plt.title(title)
    
    if wordembedding == "1":
        plt.savefig("./images/Kmeans_BoW.png", dpi=300)
    elif wordembedding == "2":
        plt.savefig("./images/Kmeans_tfidf.png", dpi=300)

    print("Saved images successfully!")
    # plt.show()
