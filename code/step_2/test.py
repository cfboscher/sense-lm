from step_2.classifier import build_model
from step_2.tools import complete_words, model_check_point_tensor_board
from step_2.metrics import get_prec, get_rec

from keras import backend as K
from sklearn.model_selection import StratifiedKFold

from time import time

import numpy as np


def test(tokenizer, model, train_data, test_data, input_ids_t, attention_mask_t, token_type_ids_t, config):

    MAX_LEN = config.tokenizer_max_len_step2
    VER = 'v0'
    DISPLAY = 1
    sensoriality_id = {'Olfactive': 1313, 'Non-Olfactive': 2430}

    # creating bunch of variable for preprocessing
    test_data = test_data[test_data['ref_type'] == 'Olfactive'].reset_index(drop="true")
    ct = test_data.shape[0]
    input_ids_t = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask_t = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids_t = np.zeros((ct, MAX_LEN), dtype='int32')

    for k in range(test_data.shape[0]):
        # INPUT_IDS
        # tokenize the test data as same as train data, we have not given the start and end token (as it is only for training)
        text1 = " " + " ".join(str(test_data.loc[k, 'clean_sentence']).split())
        enc = tokenizer.encode(text1)
        s_tok = sensoriality_id[test_data.loc[k, 'ref_type']]
        input_ids_t[k, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
        attention_mask_t[k, :len(enc.ids) + 5] = 1


    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    preds_folds = []
    true_folds = []
    mappings = []

    t_oof_start = np.zeros((input_ids_t.shape[0], MAX_LEN))
    t_oof_end = np.zeros((input_ids_t.shape[0], MAX_LEN))

    true_pred_mapping = {}
    # skf split will generate train and test data
    for fold, (idxT, idxV) in enumerate(skf.split(input_ids_t, test_data.ref_type.values)):

        if fold !=4 :
            continue
        begin = time()

        idxV = np.concatenate((idxT, idxV), axis=None)


        print('#' * 25)
        print('### FOLD %i' % (fold + 1))
        print('#' * 25)

        # clear the session
        # K.clear_session()
        # call the model
        # model = build_model(config)

        # fit the model by passing the train and validation data   which have been defined as idxT and IdxV
        # attension mask will tells roBERTa which tokens are meaningful so roBERTa can ignore the rest
        # model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]], [start_tokens[idxT,], end_tokens[idxT,]],
        #    epochs=3, batch_size=32, verbose=DISPLAY, callbacks= model_check_point_tensor_board(),
        #    validation_data=([input_ids[idxV,],attention_mask[idxV,],token_type_ids[idxV,]],
        #    [start_tokens[idxV,], end_tokens[idxV,]]))

        # print('Loading model...')
        model.load_weights('%s-roberta-%i.h5' % (VER, fold))

        print('Predicting OOF...')
        t_oof_start[idxV,], t_oof_end[idxV,] = model.predict(
            [input_ids_t[idxV,], attention_mask_t[idxV,], token_type_ids_t[idxV,]],
            verbose=DISPLAY)

        print('Predicting Test...')
        preds = model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=DISPLAY)


        true_text = []
        preds_text = []

        all_prec = []
        all_rec = []
        # go through the validation data and take the  max value from oof start and end (which has been prepicted earlier)
        for k in idxV:
            a = np.argmax(t_oof_start[k,])
            b = np.argmax(t_oof_end[k,])
            if a > b:
                st = test_data.loc[k, 'clean_sentence']
            else:
                text1 = " " + " ".join(str(test_data.loc[k, 'clean_sentence']).split())
                enc = tokenizer.encode(text1)
                st = tokenizer.decode(enc.ids[a - 1:b])

            st = complete_words(st, test_data.loc[k, 'clean_token'])

            # Fix loop to restore partially extracted words
            st_split = st.split()

            for word_2 in range(len(st_split)):

                for word in str(test_data.loc[k, 'clean_token']).split():

                    if st_split[word_2] in word:
                        st_split[word_2] = word


            pred_words = st.split()
            true_words = str(test_data.loc[k, 'clean_token']).split()
            if len(pred_words) > 0:
                all_prec.append(len([x for x in pred_words if x in true_words]) / len(pred_words))
            if len(true_words) > 0:
                all_rec.append(len([x for x in pred_words if x in true_words]) / len(true_words))

            true_text.append(test_data.loc[k, 'clean_token'])
            preds_text.append(st)
            test_data.loc[k, 'pred_token'] = st

        print('>>>> FOLD %i Precision=' % (fold + 1), np.mean(all_prec))
        print('>>>> FOLD %i Recall=' % (fold + 1), np.mean(all_rec))
        print()

        mappings.append(true_pred_mapping)
        preds_folds.append(preds_text)
        true_folds.append(true_text)

        end = time()

        print("Time Elapsed ")
        print(end - begin)
    return test_data


def eval_roberta(test_data):
    test_data['prec'] = test_data.apply(lambda x: get_prec(x.pred_token, x.clean_token), axis=1)
    test_data['rec'] = test_data.apply(lambda x: get_rec(x.pred_token, x.clean_token), axis=1)

    print("Precision after Step 2.1", test_data['prec'].mean())
    print("Recall after Step 2.1", test_data['rec'].mean())

    return test_data