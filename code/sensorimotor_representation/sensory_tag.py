from itertools import chain
from nltk.corpus import wordnet

def sensory_tag_token(token, sensoriality, sensorimotor_norms):

    if token in sensorimotor_norms:
        score = sensorimotor_norms[token][sensoriality]
    else:
        score = 0
        synonyms = wordnet.synsets(token)
        candidates = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))

        for candidate in candidates:
            if candidate in sensorimotor_norms:
                score = sensorimotor_norms[candidate][sensoriality]

    return score


def sensory_tag_sentence(sentence, sensorimotor_norms, nlp):
    s = 0

    doc = nlp(sentence)

    sensory_dict = {'AUD': 0, 'GUS': 0, 'HAP': 0, 'INT': 0, 'OLF': 0, 'VIS': 0, 'ARM': 0,
                    'LEG': 0, 'TORSO': 0, 'HEAD': 0, 'MOUTH': 0
                    }

    for token in doc:
        if token.pos_ in ['NOUN', 'ADJ', 'ADV', 'VERB']:
            sensory_dict['AUD'] += sensory_tag_token(token.lemma_, 'Auditory', sensorimotor_norms)
            sensory_dict['GUS'] += sensory_tag_token(token.lemma_, 'Gustatory', sensorimotor_norms)
            sensory_dict['HAP'] += sensory_tag_token(token.lemma_, 'Haptic', sensorimotor_norms)
            sensory_dict['INT'] += sensory_tag_token(token.lemma_, 'Interoceptive', sensorimotor_norms)
            sensory_dict['OLF'] += sensory_tag_token(token.lemma_, 'Olfactory', sensorimotor_norms)
            sensory_dict['VIS'] += sensory_tag_token(token.lemma_, 'Visual', sensorimotor_norms)

            sensory_dict['ARM'] += sensory_tag_token(token.lemma_, 'Hand_arm', sensorimotor_norms)
            sensory_dict['LEG'] += sensory_tag_token(token.lemma_, 'Foot_leg', sensorimotor_norms)
            sensory_dict['TORSO'] += sensory_tag_token(token.lemma_, 'Torso', sensorimotor_norms)
            sensory_dict['HEAD'] += sensory_tag_token(token.lemma_, 'Head', sensorimotor_norms)
            sensory_dict['MOUTH'] += sensory_tag_token(token.lemma_, 'Mouth', sensorimotor_norms)

            s += 1

    #     for key in sensory_dict.keys():
    #         if s > 0:
    #             sensory_dict[key] /= s
    return sensory_dict