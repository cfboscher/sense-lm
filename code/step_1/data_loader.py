import torch
import spacy
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from sensorimotor_representation.sensory_tag import sensory_tag_sentence

def get_data_loader(data, tokenizer, sensorimotor_norms):

    spacy_model = spacy.load("en_core_web_md")
    mapping = {False: 0, True: 1}

    sentences = data['text'].to_list()
    labels = [mapping[x] for x in data['contains_ref'].to_list()]

    max_len = 0

    # For every sentence...
    for sent in sentences:
        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    print('Max sentence length: ', max_len)

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []
    sensorimotor_embeddings = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

        # Add the sensorimotor representation of the sentence
        sensorimotor_embeddings.append(list(sensory_tag_sentence(sent, sensorimotor_norms, spacy_model).values()))

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    sensorimotor_embeddings = torch.stack([torch.tensor(sent) for sent in sensorimotor_embeddings])
    labels = torch.tensor(labels)

    # Print sentence 0, now as a list of IDs.
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

    dataset = TensorDataset(input_ids, attention_masks, sensorimotor_embeddings, labels)

    batch_size = 8

    dataloader = DataLoader(
                       dataset,  # The training samples.
                       sampler=RandomSampler(dataset), # Select batches randomly
                       batch_size=batch_size # Trains with this batch size.
                        )

    return dataloader
