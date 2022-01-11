from sklearn.feature_extraction.text import TfidfVectorizer
from utils.pipeline import Pipeline


if __name__ == '__main__':

    threshold = 1
    input_file = 'data/wos_abstracts.txt'
    output_file = 'tf_idf.txt'

    corpus = []
    with open(input_file, 'r', encoding='utf-8') as fr:
        for doc in fr:
            corpus.append(doc.strip())

    pipeline = Pipeline()
    result = pipeline.preprocess_corpus(corpus)

    # transforms Pipeline's output type (List[List[str]]) into TfidfVectorizer's input type (List[str])
    docs = []
    for doc in result:
        new_doc = []
        for sent in doc:
            for tok in sent:
                new_doc.append(tok)

        if len(new_doc) > 0:
            docs.append(' '.join(new_doc))

    vectorizer = TfidfVectorizer(min_df=3)
    vectorizer.fit(docs)

    # Transform documents to document-term matrix
    doc_term_matrix = vectorizer.transform(docs)
    doc_term_matrix = doc_term_matrix.toarray()

    # get max tfidf value per each word
    max_tfidf_vector = doc_term_matrix.max(axis=0)

    voca_dict = vectorizer.vocabulary_
    tfidf_dict = {}
    for word, word_id in voca_dict.items():
        weight = max_tfidf_vector[word_id]
        tfidf_dict[word.strip()] = weight


    # sort and store
    with open(output_file, 'w', encoding='utf-8') as fw:
        sorted_dict = sorted(tfidf_dict.items(), key=lambda x: x[1], reverse=True)

        for word, weight in sorted_dict:
            if weight > threshold:
                print(f'{word}\t{weight[1]}')
                fw.write(f'{word}\t{weight[1]}\n')
