import pandas as pd
from fastai.text import (TextClasDataBunch, text_classifier_learner, AWD_LSTM, TextLMDataBunch, language_model_learner)
from foods.settings import intermediate
magic_number = 2.6**4 


def RNN_fastai(path, file_name, version, lm_cycles=8, classifier_cycles=10, text_cols='text', 
    label_cols='label', drop_mult=0.2, encoder_exists = False):
    """Train (and save) a sentiment predictor with transfer learning. The base model is a FastAi language model
        (LSTM RNN trained on the Wikitext-103 dataset)

    Parameters
    ----------
    path : str
        path of the text dataset
    file_name : str
        name of the text dataset
    version : str
        version name
    lm_cycles : int, optional
        Number of training cycles on the unfrozen RNN for the language model, by default 3
    classifier_cycles : int, optional
        Number of training cycles on teh unfrozen RNN for the predictor, by default 8
    text_cols : str, optional
        text column name in the text dataset, by default 'text'
    label_cols : str, optional
        sentiment label column name in the text dataset, by default 'label'
    drop_mult : float, optional
        dropout ratio in the learner (RNN), by default 0.5
    encoder_exists : boolean, optional
       whether to use a pre-trained language model encoder, by default false

    Returns
    -------
    [type]
        Trained sentiment predictor
    """
    # QA on inputs
    if (lm_cycles<5) | (classifier_cycles<5):
        print("training cyles too small. Aborting...")
        return

    # Initiate and tune language model
    if not encoder_exists:
        learner_lm, data_lm = initiate_learner(path, file_name, data_lm = None, encoder_name = None, 
                                            text_cols=text_cols, label_cols=label_cols, drop_mult = drop_mult)

        learner_lm = tune_save_learner(learner_lm, 0.1, 0.003, lm_cycles, f'{version}_RNN_lm')

    # Initiate and tune classifier
    learner_c, data_lm = initiate_learner(path, file_name, data_lm = data_lm, 
                                            encoder_name = f'{version}_RNN_lm', 
                                            text_cols=text_cols, label_cols=label_cols, drop_mult = drop_mult)

    learner_c = tune_save_learner(learner_c, 0.1, 0.001, classifier_cycles, f'{version}_RNN_classifier')
  
    return learner_c


def initiate_learner(path, file_name, data_lm = None, encoder_name = None, text_cols='text', label_cols='label', drop_mult = 0.2):
    """Initiate a sentimental classifer or a language model using the text dataset under path/file_name.

    Parameters
    ----------
    path : str
        path of the text dataset
    file_name : str
        name of the text dataset
    data_lm : None or fastai.text.TextLMDataBunch, optional
        data bunch for the language model, by default None
    encoder_name : str, optional
        name of the pre-tuned language model encoder for the classifier, by default None
    text_cols : str, optional
        text column name in the text dataset, by default 'text'
    label_cols : str, optional
        sentiment label column name in the text dataset, by default 'label'
    drop_mult : float, optional
        dropout ratio in the learner (RNN), by default 0.2

    Returns
    -------
    fastai.text learner

    """
    # Initiate a sentimental classifer on top of the pre-tuned language model
    if encoder_name:
        data_clas = TextClasDataBunch.from_csv(path, file_name, text_cols=text_cols, label_cols=label_cols,
                                            vocab=data_lm.train_ds.vocab, bs=32)
        learner = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=drop_mult)
        learner.load_encoder(encoder_name)
    
    # Initiate a language model to tune. The language model is based on fastai's RNN model trained on Wikitext 103 dataset,
    # which predicts the next word for any string inputs
    else:
        data_lm = TextLMDataBunch.from_csv(path, file_name, text_cols=text_cols, label_cols=label_cols) 
        learner = language_model_learner(data_lm, AWD_LSTM, drop_mult=drop_mult)

    return learner, data_lm

def tune_save_learner(learner, initial_LR, onging_LR, training_cycles, learner_name):
    """"
    Tune the learner (either a language model or a sentiment clssifier) and save it.
    
    Parameters
    ----------
    learner : fastai.text.language_model_learner or fastai.text.TextClasDataBunch
        Input learner to tune
    initial_LR : float
        Learning rate for the last layers in the 1st cycle of training
    onging_LR : float
        Learning rate for all layers in the following cycles
    training_cycles : int
        number of training cycles
    learner_name : str
        learner name
    
    Returns
    -------
    Tuned learner
    """
    # Tune the last layer
    print(f"Train the last layer of the model...")
    learner.fit_one_cycle(1, initial_LR)
     
    # Tune the last 2-3 layers
    for i in range(2, 4): #FIXME check the number of layers in the RNN
        print(f"Train the last {i} layers of the model...")
        learner.freeze_to(-i)
        learner.fit_one_cycle(1, slice(onging_LR/magic_number, onging_LR), moms=(0.8,0.7))

    # Tune all layers
    learner.unfreeze()
    for i in range(training_cycles-4):
        print(f"Train the all layers of the model: cycle {i}")
        learner.fit_one_cycle(1, slice(onging_LR/magic_number, onging_LR), moms=(0.8,0.7))

        if learner_name.split('_')[-1] == 'lm':
            learner.save_encoder(f"{learner_name}_cycle_{i}")
        elif learner_name.split('_')[-1] == 'classifier':
            learner.export(f'export_all_{i}.pkl')
        else:
            print('Save method not defined, learner not saved')

    if learner_name.split('_')[-1] == 'lm':
        learner.save_encoder(f"{learner_name}")
    elif learner_name.split('_')[-1] == 'classifier':
        learner.export(f'export_all.pkl')
    else:
        print('Save method not defined, learner not saved')

    return learner