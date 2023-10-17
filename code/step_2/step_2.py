from utils.prepare_data import prepare_data_step_2

from sensorimotor_representation.load_sensorimotor_norms import load_sensorimotor_norms

from step_2.tokenizer import load_tokenizer, tokenize_train_data, tokenize_test_data

from step_2.train import train
from step_2.test import test, eval_roberta

from step_2.lexifield import apply_lexifield, get_lexifield_terms, eval_lexifield
from step_2.heuristic import apply_heuristic, load_glove_models, eval_heuristic

from step_2.metrics import get_prec, get_rec

from sklearn.model_selection import train_test_split


def run(config):
    print('Running Step 2')

    df = prepare_data_step_2(config.data_path, config.dataset)

    print('Splitting data...')
    train_data, test_data = train_test_split(df, test_size=0.1, random_state=config.random_seed)
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()

    print('Loading Sensorimotor Norms...')
    sensorimotor_norms = load_sensorimotor_norms('sensorimotor_representation')

    # STEP 2.1

    # Load the RoBERTa tokenizer.
    print('Loading Tokenizer...')
    tokenizer, sensoriality_id = load_tokenizer()

    print('Tokenizing Train Data')
    input_ids, attention_mask, token_type_ids, start_tokens, end_tokens = tokenize_train_data(tokenizer, train_data,
                                                                                              sensoriality_id, config)

    print('Tokenizing Test Data')
    input_ids_t, attention_mask_t, token_type_ids_t = tokenize_test_data(tokenizer, test_data, sensoriality_id, config)

    print("STEP 2.1 ")
    print('Train Model for Step 2.1')
    model = train(tokenizer,  train_data,
                              input_ids, input_ids_t,
                              attention_mask, attention_mask_t,
                              token_type_ids, token_type_ids_t, start_tokens, end_tokens,
                              config)

    print('Test Models for Step 2.1')
    test_data = test(tokenizer, model, test_data, train_data, input_ids_t, attention_mask_t,
                                                      token_type_ids_t, config)


    test_data = eval_roberta(test_data)
    print("Precision after Step 2.1", test_data['prec'].mean())
    print("Recall after Step 2.1", test_data['rec'].mean())


    # STEP 2.2
    print("STEP 2.2")
    print("Expanding predictions with Lexifield")
    lexifield_terms = get_lexifield_terms(config)
    test_data['pred_token'] = test_data.apply(
        lambda x: apply_lexifield( x.clean_sentence, x.pred_token, lexifield_terms), axis=1)

    test_data = eval_lexifield(test_data)


    # STEP 2.3
    print("STEP 2.3")
    print("Expanding predictions with Heuristic")
    print("Loading Glove Models")
    glove_models = load_glove_models()
    test_data['pred_token'] = test_data.apply(
        lambda x: apply_heuristic(x.clean_sentence, x.pred_token, sensorimotor_norms, glove_models, config), axis=1)

    test_data = eval_heuristic(test_data)
