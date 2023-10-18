import json


class Config:
    def __init__(self):
        # General Settings
        self.device = 'cuda'
        self.data_path = '../data'
        self.random_seed = 42

        self.dataset = 'auditory' # Please set either "auditory" or "odeuropa"

        self.steps = "all" # Please set either "all", "1" or "2"

        # Settings of Step 1
        self.pretrained_parameters_step1 = "bert-base-uncased" # HuggingFace URL of the Pytorch Model Parameters
        self.epochs_step1 = 20
        self.learning_rate_step1 = 2e-5
        self.epsilon_step1 = 1e-8

        # Settings of Step 2
        self.epochs_step2 = 20
        self.learning_rate_step2 = 2e-5
        self.tokenizer_max_len_step2 = 303
        self.epsilon_step2 = 1e-8
        self.T = 3.5
        self.U = 0.95





def update_config(default_config, external_config_file):
    """
    Updates the default configuration by overwriting the values if an external config file is provided

    :param default_config: Config class instance
    :param external_config_file: Experiment arguments
    :return: Updated configuration
    """

    # Parse experiment schedule JSON file
    f = open(external_config_file)
    cfg = json.load(f)

    # Update keys and values
    for key, value in cfg['config'].items():
        setattr(default_config, key, value)

    return default_config
