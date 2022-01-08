from utils.pipeline import Pipeline
import operator

if __name__ == '__main__':

    threshold = 0
    input_file = 'data/wos_abstracts.txt'
    output_file = 'tf.txt'

    # load documents on memory
    corpus = []
    with open(input_file, 'r', encoding='utf-8') as fr:
        for doc in fr:
            corpus.append(doc)

    # preprocess corpus
    pipeline = Pipeline()
    result = pipeline.preprocess_corpus(corpus)


    # count word's frequency
    term_counter = {}
    for doc in result:
        for sent in doc:
            for token in sent:
                if token not in term_counter.keys():
                    term_counter[token] = 1
                else:
                    term_counter[token] += 1

    # sort and store
    with open(output_file, 'w', encoding='utf-8') as fw:
        sorted_dict = sorted(term_counter.items(), key=operator.itemgetter(1), reverse=True)
        for term_and_freq in sorted_dict:
            if term_and_freq[1] >= threshold:
                #print(f'{term_and_freq[0].strip()}\t{term_and_freq[1]}')
                fw.write(f'{term_and_freq[0].strip()}\t{term_and_freq[1]}\n')
