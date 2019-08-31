import foods.data.inputs as F_IO
import foods.models.RNN as F_RNN
import foods.data.post_processing as F_post
import foods.settings as settings
from fastai.text import load_learner


def main(
    version="Aug9k_Drop2_LM8_CL10",
    number_of_augmentation=9000,
    drop_mult=0.6,
    encoder_exists=False,
):
    _, _, _ = F_IO.prepare_train_dev(
        dev_size=0.2,
        number_of_augmentation=number_of_augmentation,
        clean_txt=False,
        stem_option=False,
        rem_stop_option=False,
    )

    learner_c = F_RNN.RNN_fastai(
        settings.intermediate,
        settings.train_formated,
        version,
        lm_cycles=8,
        classifier_cycles=15,
        text_cols="text",
        label_cols="label",
        drop_mult=drop_mult,
        encoder_exists=encoder_exists,
    )

    return learner_c
