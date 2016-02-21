import numpy as np
import stop_words as sw

def get_word_count(path2data, path2label, target_labels=None):
    '''
    Create a word count dictionary where keys are word ids and values are counts in the data.
    '''    
    data = np.genfromtxt(path2data, dtype=int)
    labels = np.genfromtxt(path2label, dtype=int)

    vocab_count = {}
    for doc_id, word_id, word_count in data:
        if labels[doc_id-1] not in target_labels:
            continue
        
        if not vocab_count.has_key(word_id):
            vocab_count[word_id] = word_count
        else:
            vocab_count[word_id] += word_count

    return vocab_count


def merge_dicts(dic1, dic2):
    '''
    Merge two dictionaries and return the merged. 
    If two dictionaries both have a key, the merged has the key with value being the sum of the two values from dic1 and dic2.
    '''
    merged_map = {}
    for k, v in dic1.iteritems():
        merged_map[k] = v

    for k, v in dic2.iteritems():
        if merged_map.has_key(k):
            merged_map[k] += v
        else:
            merged_map[k] = v
    
    return merged_map


def get_top_words(vocabmap, path2vocabs, k=50, ignore_stopwords=True, swords=sw.get_stop_words('english')):
    '''
    Get top k (frequent) words from a word-count dictionary.
    Ids and the corresponding words in a given list of vocabulary are returned.
    '''
    vocabs_counts = vocabmap.items()
    vocabs_counts.sort(key=lambda x: x[1], reverse=True)
    vocabs = np.genfromtxt(path2vocabs, dtype=str)

    top_wids, top_words = [], []
    nc = 0
    for wid, c in vocabs_counts:
        if not ignore_stopwords or vocabs[wid-1] not in swords:
            top_wids.append(wid)
            top_words.append(vocabs[wid-1])
            nc += 1
        if nc >= k:
            break

    return top_words, top_wids


def get_docmat(wids, path2data, path2label, target_labels=None):
    ''' 
    Create a document-term 2d array where each row corresponds to a document and each column corresponds to a word in the given wids (in the same order).
    '''
    wid2idx = {}
    for i, wid in enumerate(wids):
        wid2idx[wid] = i

    data = np.genfromtxt(path2data, dtype=int)
    labels = np.genfromtxt(path2label, dtype=int)

    out = np.zeros((np.max(data[:, 0]), len(wids)))
    ndocs = 0    
    last_doc_id = 0

    for doc_id, word_id, word_count in data:
        if word_id not in wids:
            continue

        if labels[doc_id-1] not in target_labels:
            continue
        
        if doc_id != last_doc_id:
            ndocs += 1
        
        out[ndocs-1, wid2idx[word_id]] += word_count
        last_doc_id = doc_id    

    return out[:ndocs]

