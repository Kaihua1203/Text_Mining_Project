from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt

def run_tsne(data_text, data_label, wordembedding):
    if wordembedding == "1":  # Bag of Words
        vectorizer = CountVectorizer()
        data_text_vectorized = vectorizer.fit_transform(data_text)
        title = 't-SNE Visualization with Bag of Words'

    elif wordembedding == "2":  # TF-IDF
        vectorizer = TfidfVectorizer()
        data_text_vectorized = vectorizer.fit_transform(data_text)
        title = 't-SNE Visualization with TF-IDF'

    else:
        print("Invalid input. Please enter 1 or 2.")
        return

    X_tsne = TSNE(n_components=2, random_state=42, init='random').fit_transform(data_text_vectorized)

    plt.figure(figsize=(10, 5))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=data_label, cmap='viridis', alpha=0.6)
    plt.title(title)
    plt.colorbar(label='Categories', ticks=range(len(set(data_label))))
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.tight_layout()

    # Save the plot as a high-resolution image
    if wordembedding == "1":
        plt.savefig("./images/Tsne_BoW.png", dpi=300)
    elif wordembedding == "2":
        plt.savefig("./images/Tsne_tfidf.png", dpi=300)

    print("Saved images successfully!")

    # Show the plot
    plt.show()
