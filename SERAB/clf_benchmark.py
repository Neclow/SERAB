# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate a pre-trained model on the Speech Emotion Recognition Adaptation Benchmark (SERAB).

The following functions were inspired by: https://github.com/google-research/google-research/tree/master/non_semantic_speech_benchmark
@article{shor2020,
    title={Towards Learning a Universal Non-Semantic Representation of Speech},
    author={Joel Shor and Aren Jansen and Ronnie Maor and Oran Lang and Omry Tuval and Felix de Chaumont Quitry and Marco Tagliasacchi and Ira Shavitt and Dotan Emanuel and Yinnon Haviv},
    year={2020},
    journal = {ArXiv e-prints},
    eprint={2002.12764},
    archivePrefix={arXiv},
    primaryClass={eess.AS},
    url = {https://arxiv.org/abs/2002.12764}
}

"""

import collections
import os

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import torch

from pytorch_lightning.utilities.seed import seed_everything
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from torchaudio.transforms import MelSpectrogram

from byol_a.common import load_yaml_config
from byol_a.augmentations import PrecomputedNorm
from settings import CLF_STATS_DICT, RANDOM_SEED, REQUIRED_SAMPLE_RATE
from utils import compute_norm_stats, generate_embeddings, load_model, save_results, speaker_normalization


def parse_args():
    """Parse main arguments."""
    parser = ArgumentParser(description='benchmark arguments')

    parser.add_argument(
        '-model_name', '--model_name',
        type=str, default='default',
        help='Model name'
    )

    parser.add_argument(
        '-dataset_name', '--dataset_name',
        type=str, default='crema_d',
        help='Dataset name'
    )

    parser.add_argument(
        '-model_selection', '--model_selection',
        type=str, default='predefined',
        help='Model selection mode (For now: no model selection or via grid search on predefined splits)'
    )

    parser.add_argument(
        '-config_path', '--config_path',
        type=str, default='config.yaml',
        help='Path to BYOL-A configuration file'
    )

    return parser.parse_args()


def dat_from_split(tfds_dataset_name, split):
    """Load a tensorflow dataset from a split (train, valid, test).

    Adapted from https://github.com/google-research/google-research/tree/master/non_semantic_speech_benchmark

    Parameters
    ----------
    tfds_dataset_name : str
        Tensorflow dataset name
    split : str
        Dataset split, can be 'train', 'validation' or 'test'.

    Returns
    -------
    audio : list
        List of audios, loaded as a numpy arrays with dtype np.int16.
    labels : numpy.ndarray
        Array of labels for each audio.
    speaker_id : Union[numpy.ndarray, None]
        If the dataset contains speaker information, an array with the speaker ID for each audio, otherwise None.
    """

    np_generator = tfds.as_numpy(tfds.load(tfds_dataset_name, split=split))

    dat = [(x['audio'], x['label'], x['speaker_id']) for x in np_generator]
    audio, labels, speaker_id = zip(*dat)

    labels = np.array(labels, dtype=np.int16)
    speaker_id = np.array(speaker_id)
    assert len(audio) == labels.size == speaker_id.size
    assert labels.ndim == 1 == speaker_id.ndim == 1
    print(f'Finished {split}')
    return audio, labels, speaker_id


def get_sklearn_models():
    """Load a family of standard ML classifiers for downstream classification

    Current models: LDA, Logistic Regression, QDA, Random Forests, SVC

    Models are loaded as a dict with a model-specific parameter grid for hyperparameter optimization.

    Returns
    -------
    log_list : list
        List of estimator names
    estimator_list : list
        List of estimator objects
    param_list : list
        List of estimator parameter grids
    """

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    params_lda = {
        'solver': ['lsqr']
    }

    lr = LogisticRegression(class_weight='balanced', max_iter=1e4, random_state=RANDOM_SEED)
    params_lr = {
        'C': np.logspace(5, -5, num=11),
        'class_weight': ['balanced', None],
    }

    qda = QuadraticDiscriminantAnalysis()
    params_qda = {
        'reg_param': [0, 0.1, 0.25]
    }

    rf = RandomForestClassifier(class_weight='balanced', random_state=RANDOM_SEED)
    params_rf = {
        'max_depth': range(5, 30, 5),
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', None],
    }

    svc = SVC(max_iter=1e4, random_state=RANDOM_SEED)
    params_svc = {
        'kernel': ['rbf', 'linear'],
        'C': np.logspace(5, -5, num=11),
        'class_weight': ['balanced', None]
    }

    estimator_list = [lda, lr, qda, rf, svc]
    log_list = ['LDA', 'LR', 'QDA', 'RF', 'SVC']
    param_list = [params_lda, params_lr, params_qda, params_rf, params_svc]

    return log_list, estimator_list, param_list


if __name__ == '__main__':
    # Load config
    args = parse_args()
    cfg = load_yaml_config(args.config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(RANDOM_SEED)

    to_melspec = MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
    )

    assert args.model_selection in ['predefined', 'none'], 'Model selection mode not found'

    # Load data
    SingleSplit = collections.namedtuple('SingleSplit', ['audio', 'labels', 'speaker_id'])
    Data = collections.namedtuple('Data', ['train', 'validation', 'test'])
    all_data = Data(
        train=SingleSplit(*dat_from_split(args.dataset_name, 'train')),
        validation=SingleSplit(*dat_from_split(args.dataset_name, 'validation')),
        test=SingleSplit(*dat_from_split(args.dataset_name, 'test')))

    orig_sr = tfds.builder(args.dataset_name).info.features['audio'].sample_rate
    num_classes = len(np.unique(all_data.train.labels))

    # Load data statistics
    try:
        stats = CLF_STATS_DICT[args.dataset_name]
    except KeyError:
        print(f'Did not find mean/std stats for {args.dataset_name}.')
        stats = compute_norm_stats(args.dataset_name, all_data.train.audio, orig_sr, to_melspec)

        CLF_STATS_DICT[args.dataset_name] = stats

        print(CLF_STATS_DICT)
    normalizer = PrecomputedNorm(stats)

    # Load model and disable parameter training
    model, weight_file = load_model(args.model_name, cfg, device)

    # Generate embeddings
    embeddings = Data(
        train=generate_embeddings(
            model,
            args.model_name,
            all_data.train.audio,
            'train',
            orig_sr,
            to_melspec,
            normalizer,
            device
        ),
        validation=generate_embeddings(
            model,
            args.model_name,
            all_data.validation.audio,
            'validation',
            orig_sr,
            to_melspec,
            normalizer,
            device
        ),
        test=generate_embeddings(
            model,
            args.model_name,
            all_data.test.audio,
            'test',
            orig_sr,
            to_melspec,
            normalizer,
            device
        )
    )

    print(embeddings.train.mean(), embeddings.train.std())
    print(embeddings.validation.mean(), embeddings.validation.std())
    print(embeddings.test.mean(), embeddings.test.std())
    print(embeddings.train.shape)
    print(embeddings.validation.shape)
    print(embeddings.test.shape)

    # Load classifiers
    log_list, estimator_list, param_list = get_sklearn_models()

    # Speaker normalization
    # Can also try with normal standardization (StandardScaler), should not change the results too much
    normalized_train = speaker_normalization(embeddings.train, all_data.train.speaker_id)
    normalized_validation = speaker_normalization(embeddings.validation, all_data.validation.speaker_id)
    normalized_test = speaker_normalization(embeddings.test, all_data.test.speaker_id)

    # Aggregate labels and speaker IDs
    normalized_train = np.append(normalized_train, normalized_validation, axis=0)
    labels_train = np.append(all_data.train.labels, all_data.validation.labels, axis=0)
    speaker_id_train = np.append(all_data.train.speaker_id, all_data.validation.speaker_id, axis=0)

    # Collect performance data for each estimator
    results = {}
    for i, (estimator_name, estimator, param_grid) in enumerate(zip(log_list, estimator_list, param_list)):
        print(f'Step {i+1}/{len(estimator_list)}: {estimator_name}...')
        if args.model_selection == 'predefined':
            split_indices = np.repeat([-1, 0], [embeddings.train.shape[0], embeddings.validation.shape[0]])
            split = PredefinedSplit(split_indices)
            clf = GridSearchCV(
                estimator,
                param_grid,
                cv=split,
                n_jobs=-1,
                refit=True,
                verbose=0
            )

        else:
            clf = estimator

        clf.fit(normalized_train, labels_train)
        test_acc = clf.score(normalized_test, all_data.test.labels)
        test_uar = recall_score(all_data.test.labels, clf.predict(normalized_test), average='macro')

        results[estimator_name] = {
            'test_acc': test_acc,
            'test_uar': test_uar
        }

        print('Done')

    # Collect test accuracy results
    results_df = pd.DataFrame(results).apply(lambda x: round(x * 100, 1))

    results_df = results_df[results_df.loc['test_acc'].idxmax()].to_frame()
    print(results_df)
    results_df = results_df.rename(columns={results_df.columns[0]: args.dataset_name})

    # Save results
    filename = os.path.splitext(os.path.basename(weight_file))[0] if weight_file else args.model_name
    results_folder = 'clf_results/'
    save_results(filename + '.csv', results_df, results_folder)
