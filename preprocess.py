import utils
import numpy as np
import cPickle as pickle

hardware_label, baseball_label, religion_label = 4, 10, 20

# get the most frequent 50 non-stop words from each newsgroup
# note: train and test were for classification task and irrelevant for our purpose and thus we simply merge them
path2dir = '20news-bydate-matlab/matlab/'

train_vocabmap = utils.get_word_count(path2dir + 'train.data', path2dir + 'train.label', [hardware_label])
test_vocabmap = utils.get_word_count(path2dir + 'test.data', path2dir + 'test.label', [hardware_label])
htop50w, htop50id = utils.get_top_words(utils.merge_dicts(train_vocabmap, test_vocabmap), 'vocabulary.txt')

train_vocabmap = utils.get_word_count(path2dir + 'train.data', path2dir + 'train.label', [baseball_label])
test_vocabmap = utils.get_word_count(path2dir + 'test.data', path2dir + 'test.label', [baseball_label])
btop50w, btop50id = utils.get_top_words(utils.merge_dicts(train_vocabmap, test_vocabmap), 'vocabulary.txt')

train_vocabmap = utils.get_word_count(path2dir + 'train.data', path2dir + 'train.label', [religion_label])
test_vocabmap = utils.get_word_count(path2dir + 'test.data', path2dir + 'test.label', [religion_label])
rtop50w, rtop50id = utils.get_top_words(utils.merge_dicts(train_vocabmap, test_vocabmap), 'vocabulary.txt')


# find the union of all word ids; this is less than 150.
allwids = list(set(htop50id).union(set(btop50id)).union(set(rtop50id)))

# create document term (count) matrices from our choice of words
hdocmat_train = utils.get_docmat(allwids, path2dir + 'train.data', path2dir + 'train.label', [hardware_label])
hdocmat_test = utils.get_docmat(allwids, path2dir + 'test.data', path2dir + 'test.label', [hardware_label])
hdocmat = np.concatenate([hdocmat_train, hdocmat_test])
np.savetxt('hardware_data.txt', hdocmat, fmt='%d')

bdocmat_train = utils.get_docmat(allwids, path2dir + 'train.data', path2dir + 'train.label', [baseball_label])
bdocmat_test = utils.get_docmat(allwids, path2dir + 'test.data', path2dir + 'test.label', [baseball_label])
bdocmat = np.concatenate([bdocmat_train, bdocmat_test])
np.savetxt('baseball_data.txt', bdocmat, fmt='%d')

rdocmat_train = utils.get_docmat(allwids, path2dir + 'train.data', path2dir + 'train.label', [religion_label])
rdocmat_test = utils.get_docmat(allwids, path2dir + 'test.data', path2dir + 'test.label', [religion_label])
rdocmat = np.concatenate([rdocmat_train, rdocmat_test])
np.savetxt('religion_data.txt', rdocmat, fmt='%d')


