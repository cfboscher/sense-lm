import tokenizers
import numpy as np

def load_tokenizer():
    # reference of this transformer model : https://www.kaggle.com/cdeotte/tensorflow-roberta-0-705
    # reference text preprocessing  : https://www.youtube.com/watch?v=XaQ0CBlQ4cY

    # creating a bytelevel tokenizer
    # this tokenizer has "" ids, type_ids, tokens, offsets, attention_mask, special_tokens_mask, overflowing "" these attributes
    # the byte level BPE tokenizer done something like subword tokenizer where it will break a signle word into two word , example: faster could be fast and ##er
    tokenizer = tokenizers.ByteLevelBPETokenizer(vocab='step_2/roberta-base/vocab-roberta-base.json',
                                                 merges='step_2/roberta-base/merges-roberta-base.txt',
                                                 lowercase=True,
                                                 add_prefix_space=True)

    sensoriality_id = {'Olfactive': 1313, 'Non-Olfactive': 2430}

    return tokenizer, sensoriality_id


def tokenize_train_data(tokenizer, train_data, sensoriality_id, config):

    MAX_LEN = config.tokenizer_max_len_step2

    # creating bunch of variable for preprocessing
    ct = train_data.shape[0]
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')
    start_tokens = np.zeros((ct, MAX_LEN), dtype='int32')
    end_tokens = np.zeros((ct, MAX_LEN), dtype='int32')

    for k in range(ct):

        # FIND OVERLAP
        # take the text and selected text and bring them in a uniform manner (sentences)
        text1 = " " + " ".join(str(train_data.loc[k, 'clean_sentence']).split())
        text2 = " ".join(str(train_data.loc[k, 'clean_token']).split())
        # if text2 has all the word which is in text1 it will return 0 otherwise -1
        idx = text1.find(text2)
        # create zero vector size of the length
        chars = np.zeros((len(text1)))
        chars[idx:idx + len(text2)] = 1
        if text1[idx - 1] == ' ': chars[idx - 1] = 1
        enc = tokenizer.encode(text1)

        # ID_OFFSETS
        # offsets : create a bunch of sets ,put word  start and end number in a particular set
        # here is sentence : "understanding offset and idx"
        # the offsets are: [(0, 13), (13, 20), (20, 24), (24, 27), (27, 28)]
        # idx :  basically the end number of the word
        # the idx is : 28
        offsets = [];
        idx = 0
        for t in enc.ids:
            w = tokenizer.decode([t])
            offsets.append((idx, idx + len(w)))
            idx += len(w)

        # START END TOKENS
        # define a toks which will store  which basically store number from text2
        toks = []
        # go through the offsets
        for i, (a, b) in enumerate(offsets):
            # sum them like if there are [(2,5),(5,7)] sum them like : 3 , 2 etc
            sm = np.sum(chars[a:b])
            # now if sum is greater than zero append to tok
            # basically sm will be greater than zero where we have same text as selected text
            if sm > 0: toks.append(i)

            # create a variable to store the sentiment of the particular sentence
        s_tok = sensoriality_id[train_data.loc[k, 'ref_type']]
        # now in the input id which is a size of (27481, 96) repalce some value to : start with zero then put the encoded token
        # then 2 , 2 then the sentiment number and then 2
        #  example : lets take a sentence : "understanding offset and idx" :
        # and the encoded numbers are
        # now put :  [0 ,2969, 6147, 8, 13561, 1178,2,2 ,7974 ,2 ,.....upto the size (27481, 96) ]
        input_ids[k, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
        # create attention mask and put 1 to the len of encoded number plus 1
        attention_mask[k, :len(enc.ids) + 5] = 1

        # now if the toks is greater than zero go to the start token variabe and give 1 to the inital token and give 1 to the end token
        if len(toks) > 0:
            start_tokens[k, toks[0] + 1] = 1
            end_tokens[k, toks[-1] + 1] = 1

    return input_ids, attention_mask, token_type_ids, start_tokens, end_tokens



def tokenize_test_data(tokenizer, test_data, sensoriality_id, config):

    MAX_LEN = config.tokenizer_max_len_step2

    ct = test_data.shape[0]
    input_ids_t = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask_t = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids_t = np.zeros((ct, MAX_LEN), dtype='int32')

    for k in range(test_data.shape[0]):
        # INPUT_IDS
        # tokenize the test data as same as train data, we have not given the start and end token (as it is only for training)
        text1 = " " + " ".join(test_data.loc[k, 'clean_sentence'].split())
        enc = tokenizer.encode(text1)
        s_tok = sensoriality_id[test_data.loc[k, 'ref_type']]
        input_ids_t[k, :len(enc.ids) + 5] = [0] + enc.ids + [2, 2] + [s_tok] + [2]
        attention_mask_t[k, :len(enc.ids) + 5] = 1

    return input_ids_t, attention_mask_t, token_type_ids_t
