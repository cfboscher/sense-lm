import json
import copy

from step_2.metrics import get_prec, get_rec

def get_lexifield_terms(config):
    if config.dataset == 'odeuropa':
        path = 'step_2/lexifield/olf.json'

    elif config.dataset == 'auditory':
        path = 'step_2/lexifield/sound.json'

    with open(path) as json_file:
        json_data = json.load(json_file)

    return json_data['words']


def apply_lexifield(original_text, extracted_text, lexifield_terms):

    original_words = str(original_text).split()
    extracted_words = str(extracted_text).split()
    return_words = copy.deepcopy(extracted_words)

    for original_word in original_words:

        if original_word in lexifield_terms:
            return_words.append(original_word)

    return_words = ' '.join(return_words)

    return return_words


def eval_lexifield(test_data):
    test_data['enriched_prec'] = test_data.apply(lambda x: get_prec(x.pred_token, x.clean_token, ), axis=1)
    test_data['enriched_rec'] = test_data.apply(lambda x: get_rec(x.pred_token, x.clean_token, ), axis=1)

    print("Precision after Step 2.2", test_data['enriched_prec'].mean())
    print("Recall after Step 2.2", test_data['enriched_rec'].mean())
    return test_data