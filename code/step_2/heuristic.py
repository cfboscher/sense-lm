import copy
import gensim.downloader

from itertools import chain
from nltk.corpus import wordnet

from numpy.linalg import norm
import numpy as np

from step_2.metrics import get_prec, get_rec

from sensorimotor_representation.sensory_tag import sensory_tag_sentence_wo_pos

def apply_heuristic(original_text, extracted_text,  sensorimotor_norms, glove_models_list, config):


    dist_threshold = 0.65

    original_words = str(original_text).split()
    extracted_words = str(extracted_text).split()
    return_words = copy.deepcopy(extracted_words)


    if config.dataset == 'odeuropa':
        dim = "OLF"
    elif config.dataset == 'auditory':
        dim = 'AUD'

    extracted_words_representations = []

    for word in extracted_words:
        values = list(sensory_tag_sentence_wo_pos(word, sensorimotor_norms).values())
        if values != [0] * 11:
            extracted_words_representations.append(values)

    for original_word in original_words:

        if original_word not in extracted_words:
            sensorimotor_representation = sensory_tag_sentence_wo_pos(original_word, sensorimotor_norms)

            for extracted_word in extracted_words_representations:
                if cosine_sim(list(sensorimotor_representation.values()), extracted_word) >= config.U and \
                        sensorimotor_representation[dim] >= config.T:
                    return_words.append(original_word)

            for glove_model in glove_models_list:

                # Run heuristic for Glove Embeddings
                for extracted_word in extracted_words:
                    try:
                        if get_glove_distance(original_word, extracted_word, glove_model) >= config.U and \
                                sensorimotor_representation[dim] >= config.T:
                            return_words.append(original_word)
                    except:
                        continue

    return_words = ' '.join(return_words)

    return return_words


def load_glove_models():
    glove_models_list = ['word2vec-google-news-300', 'glove-wiki-gigaword-300']

    for i in range(len(glove_models_list)):
        glove_models_list[i] = gensim.downloader.load(glove_models_list[i])

    return glove_models_list


def get_glove_distance(original_word, extracted_word, glove_vectors):
    if original_word not in glove_vectors:

        synonyms = wordnet.synsets(original_word)
        candidates = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))

        for candidate in candidates:
            if candidate in glove_vectors:
                original_word = candidate
                break

    if extracted_word not in glove_vectors:

        synonyms = wordnet.synsets(extracted_word)
        candidates = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))

        for candidate in candidates:
            if candidate in glove_vectors:
                extracted_word = candidate
                break

    dist = glove_vectors.distance(original_word, extracted_word)

    return 1-dist

def cosine_sim(A, B):
    cosine = np.dot(A,B)/(norm(A)*norm(B))
    return cosine

def eval_heuristic(test_data):

    test_data['enriched_prec'] = test_data.apply(lambda x: get_prec(x.pred_token, x.clean_token, ), axis=1)
    test_data['enriched_rec'] = test_data.apply(lambda x: get_rec(x.pred_token, x.clean_token, ), axis=1)

    print("Precision after Step 2.3", test_data['enriched_prec'].mean())
    print("Recall after Step 2.3", test_data['enriched_rec'].mean())
    return test_data