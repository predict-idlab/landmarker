import torch
import os
from landmarker.train.utils import EarlyStopping, SaveBestModel, set_seed


def test_EarlyStopping():
    es = EarlyStopping(patience=3, min_delta=0.2, verbose=False,
                       greater_is_better=False, name_score='Val Loss')
    val_scores = [1.0, 0.7, 0.72, 0.6, 0.6, 0.795]

    for score in val_scores:
        es(score)

    assert es.early_stop == True
    assert es.best_score == 0.7

    es = EarlyStopping(patience=3, min_delta=0.0, verbose=True,
                       greater_is_better=False, name_score='Val Loss')
    val_scores = [1.0, 0.8, 0.4, 0.7, 0.6, 0.5]

    for score in val_scores:
        es(score)

    assert es.early_stop == True
    assert es.best_score == 0.4

    es = EarlyStopping(patience=3, min_delta=0.0, verbose=False,
                       greater_is_better=True, name_score='Val Loss')
    val_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.55, 0.59]

    for score in val_scores:
        es(score)

    assert es.early_stop == True
    assert es.best_score == 0.6

    es = EarlyStopping(patience=3, min_delta=0.2, verbose=False,
                       greater_is_better=True, name_score='Val Loss')
    val_scores = [0.1, 0.4, 0.8, 0.88, 0.85, 0.9]

    for score in val_scores:
        es(score)

    assert es.early_stop == True
    assert es.best_score == 0.8

    es = EarlyStopping(patience=3, min_delta=0.2, verbose=False,
                       greater_is_better=True, name_score='Val Loss')
    val_scores = [0.1, 0.4, 0.8, 0.88, 0.85, 0.9]

    for score in val_scores:
        es(score)

    assert es.early_stop == True
    assert es.best_score == 0.8


def test_SaveBestModel():
    val_scores = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    model = torch.nn.Linear(10, 1)
    sbm = SaveBestModel(verbose=True, greater_is_better=False, name_score='Val_Loss')

    for score in val_scores:
        sbm(score, model)

    assert os.path.exists(sbm.path)

    # Remove saved model
    os.remove(sbm.path)
