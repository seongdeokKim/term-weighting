from utils.pipeline import Pipeline
from utils.co_occur2graphml import convert_co_occur_to_graphml


class VocabDict:
    def __init__(self):
        self.d = {}  # word -> word_id
        self.w = []  # word_id -> word

    def get_id_or_add(self, word):
        # return word's id corresponding to that word if the word is in vocabulary
        if word in self.d:
            return self.d[word]
        # otherwise, put the word in vocabulary and then return newly generated word_id
        else:
            self.d[word] = len(self.d)
            self.w.append(word)
            return self.d[word]

    def get_id(self, word):
        # return word_id corresponding to that word if the word is in vocabulary
        if word in self.d:
            return self.d[word]
        # otherwise, return -1
        # which means that word is not in vocabulary
        return -1

    def get_word(self, id):
        return self.w[id]


if __name__ == '__main__':

    word_thres = 2
    pair_thres = 2
    input_file = 'data/wos_abstracts.txt'
    output_file = 'co_occurence.txt'


    # load documents on memory
    corpus = []
    with open(input_file, 'r', encoding='utf-8') as fr:
        for doc in fr:
            corpus.append(doc)

    # preprocess corpus
    pipeline = Pipeline()
    result = pipeline.preprocess_corpus(corpus)

    # transform the format of preprocessed corpus
    documents = []
    for doc in result:
        new_document = []
        for sent in doc:
            for token in sent:
                new_document.append(token)
        documents.append(' '.join(new_document))

    vDict = VocabDict()
    vDictFiltered = VocabDict()

    # compute word frequency
    word_counter = {}
    for document in documents:
        words = set(document.split())

        for w in words:
            wid = vDict.get_id_or_add(w)

            if wid not in word_counter:
                word_counter[wid] = 1
            else:
                word_counter[wid] += 1

    # store words that occur more than threshold in vDictFiltered
    for wid, num in word_counter.items():
        if num > word_thres:
            vDictFiltered.get_id_or_add(
                vDict.get_word(wid)
            )

    vDict = None

    # compute co-occurence within a document
    co_occur_counter = {}
    for document in documents:
        words = set(document.split())
        # convert word to word_id for computational cost
        # words that are not in vDictFiltered are excluded
        wids = list(filter(lambda wid: wid >= 0, [vDictFiltered.get_id(w) for w in words]))

        for i in range(len(wids)):
            for j in range(i+1, len(wids)):
                if wids[i] == wids[j]:
                    continue

                if wids[i] > wids[j]:
                    pair = (wids[j], wids[i])
                else:
                    pair = (wids[i], wids[j])

                if pair not in co_occur_counter:
                    co_occur_counter[pair] = 1
                else:
                    co_occur_counter[pair] += 1

    # sort and store co-occurrence info.
    with open(output_file, 'w', encoding='utf-8') as fw:
        sorted_dict = sorted(co_occur_counter.items(), key=lambda x: x[1], reverse=True)
        for (w1_id, w2_id), freq in sorted_dict:
            if freq > pair_thres:
                w1 = vDictFiltered.get_word(w1_id)
                w2 = vDictFiltered.get_word(w2_id)

                fw.write(f'{w1}\t{w2}\t{freq}\n')


    # convert co-occurence file to graphml file for further visualization through gephi software
    convert_co_occur_to_graphml(output_file, threshold=3)
