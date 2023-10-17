from step_2.classifier import build_model
from step_2.tools import complete_words, model_check_point_tensor_board

from keras import backend as K
from sklearn.model_selection import StratifiedKFold

from time import time

import numpy as np


def train(tokenizer, train_data,
          input_ids, input_ids_t,
          attention_mask, attention_mask_t,
          token_type_ids, token_type_ids_t, start_tokens, end_tokens,
          config):

    MAX_LEN = config.tokenizer_max_len_step2
    VER = 'v0'
    DISPLAY = 1

    oof_start = np.zeros((input_ids.shape[0], MAX_LEN))
    oof_end = np.zeros((input_ids.shape[0], MAX_LEN))
    preds_start = np.zeros((input_ids_t.shape[0], MAX_LEN))
    preds_end = np.zeros((input_ids_t.shape[0], MAX_LEN))

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    true_pred_mapping_train = {}
    # skf split will generate train and test data
    for fold, (idxT, idxV) in enumerate(skf.split(input_ids, train_data.ref_type.values)):

        s = time()

        idxT = idxT[:450]
        print('#' * 25)
        print('### FOLD %i' % (fold + 1))
        print('#' * 25)
        # clear the session
        K.clear_session()
        # call the model
        model = build_model(config)

        # fit the model by passing the train and validation data   which have been defined as idxT and IdxV
        # attension mask will tells roBERTa which tokens are meaningful so roBERTa can ignore the rest
        model.fit([input_ids[idxT,], attention_mask[idxT,], token_type_ids[idxT,]],
                  [start_tokens[idxT,], end_tokens[idxT,]],
                  epochs=config.epochs_step2, batch_size=8, verbose=DISPLAY, callbacks=model_check_point_tensor_board(fold),
                  validation_data=([input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]],
                                   [start_tokens[idxV,], end_tokens[idxV,]]))

        print('Loading model...')
        model.load_weights('%s-roberta-%i.h5' % (VER, fold))

        print('Predicting OOF...')
        oof_start[idxV,], oof_end[idxV,] = model.predict(
            [input_ids[idxV,], attention_mask[idxV,], token_type_ids[idxV,]], verbose=DISPLAY)

        print('Predicting Test...')
        preds = model.predict([input_ids_t, attention_mask_t, token_type_ids_t], verbose=DISPLAY)

        # The variables preds_start and preds_end are the predictions for the test set.
        # Each of the 5 folds predicts the entire test set.
        # Therefore we need to take average (fold1 + fold2 + fold3 + fold4 + fold5) / 5.0
        preds_start += preds[0] / skf.n_splits
        preds_end += preds[1] / skf.n_splits

        # DISPLAY FOLD JACCARD
        all = []
        all_cos = []
        all_prec = []
        all_rec = []

        # go through the validation data and take the  max value from oof start and end (which has been prepicted earlier)
        for k in idxV:
            a = np.argmax(oof_start[k,])
            b = np.argmax(oof_end[k,])
            if a > b:
                st = ""
            else:
                text1 = " " + " ".join(train_data.loc[k, 'clean_sentence'].split())
                enc = tokenizer.encode(text1)
                st = tokenizer.decode(enc.ids[a - 1:b])

            st = complete_words(st, train_data.loc[k, 'clean_sentence'])


            pred_words = st.split()
            true_words = str(train_data.loc[k, 'clean_token']).split()

            if len(pred_words) > 0:
                all_prec.append(len([x for x in pred_words if x in true_words]) / len(pred_words))
            if len(true_words) > 0:
                all_rec.append(len([x for x in pred_words if x in true_words]) / len(true_words))
            true_pred_mapping_train[train_data.loc[k, 'clean_token']] = st

        print('>>>> FOLD %i Precision=' % (fold + 1), np.mean(all_prec))
        print('>>>> FOLD %i Recall=' % (fold + 1), np.mean(all_rec))

        print()

        e = time()

        print(s - e)
    return model