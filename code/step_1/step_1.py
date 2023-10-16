from utils.prepare_data import prepare_data

from step_1.data_loader import get_data_loader
from step_1.classifier import SenseLM_BinaryClassifier
from step_1.train import train

from sensorimotor_representation.load_sensorimotor_norms import load_sensorimotor_norms

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import get_linear_schedule_with_warmup

import numpy as np


def run(config):

    df = prepare_data(config.data_path, config.dataset)

    print('Splitting data...')
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=config.random_seed)

    # Load the BERT tokenizer.
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_parameters_step1, do_lower_case=True)

    print('Loading Sensorimotor Norms...')
    sensorimotor_norms = load_sensorimotor_norms('sensorimotor_representation')

    print('Setting Data Loaders')
    train_dataloader = get_data_loader(train_data, tokenizer, sensorimotor_norms)
    test_dataloader = get_data_loader(test_data, tokenizer, sensorimotor_norms)

    print("Loading Classifier")
    model = SenseLM_BinaryClassifier.from_pretrained(
        config.pretrained_parameters_step1,  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=2,  # The number of output labels--2 for binary classification.
        # You can increase this for multi-class tasks.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=True,  # Whether the model returns all hidden-states.
    )

    # Tell pytorch to run this model on the GPU.
    model.cuda()

    optimizer = AdamW(model.parameters(),
                      lr=config.learning_rate_step1,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=config.learning_rate_step2  # args.adam_epsilon  - default is 1e-8
                      )

    epochs = config.epochs_step1

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)


    print("Training Classifier...")
    model, preds, true = train(model, optimizer, scheduler, train_dataloader, test_dataloader, config)

    print("Evaluation")
    preds = np.concatenate([np.argmax(i, axis=1).flatten() for i in preds])
    true = np.concatenate(true)

    precision = precision_score(true, preds, average="macro", pos_label=1)
    recall = recall_score(true, preds, average="macro", pos_label=1)

    print('Precision: ', precision)
    print('Recall: ', recall)