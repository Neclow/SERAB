"""4-class IEMOCAP dataset."""

import ast
import collections
import os

from glob import glob

import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

# Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
The Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is an acted, multimodal and multispeaker database, recently collected at SAIL lab at USC.

It contains approximately 12 hours of audiovisual data, including video, speech, motion capture of face, text transcriptions.

It consists of dyadic sessions where actors perform improvisations or scripted scenarios, specifically selected to elicit emotional expressions.

In previous research, IEMOCAP has often been evaluated as a four-class problem (Anger/Happiness/Sadness/Neutral). The resulting subset is called IEM4.

Use cases of IEM4:
* https://isca-speech.org/archive/Interspeech_2020/pdfs/3007.pdf
* https://doi.org/10.1016/j.bspc.2020.101894
* https://doi.org/10.1109/ICASSP.2016.7472790
"""

# BibTeX citation
_CITATION = """
@article{busso2008iemocap,
  title={IEMOCAP : Interactive emotional dyadic motion capture database},
  author={Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and Kazemzadeh, Abe and Mower, Emily and Kim, Samuel and Chang, Jeannette N and Lee, Sungbok and Narayanan, Shrikanth S},
  journal={Language Resources and Evaluation},
  volume={42},
  number={4},
  pages={335--359},
  year={2008},
  publisher={Springer}
}
"""

_HOMEPAGE = 'https://sail.usc.edu/iemocap/index.html'

_LABEL_MAP = {
    'ang': 'anger',
    'hap': 'happiness',
    'exc': 'happiness',
    'sad': 'sadness',
    'neu': 'neutral',
}

_SAMPLE_RATE = 16000

def parse_name(name, from_i, to_i, mapping=None):
    """Source: https://audeering.github.io/audformat/emodb-example.html"""
    key = name[from_i:to_i]
    return mapping[key] if mapping else key


def _compute_split_boundaries(split_probs, n_items):
    """Computes boundary indices for each of the splits in split_probs.
    Args:
      split_probs: List of (split_name, prob), e.g. [('train', 0.6), ('dev', 0.2),
        ('test', 0.2)]
      n_items: Number of items we want to split.
    Returns:
      The item indices of boundaries between different splits. For the above
      example and n_items=100, these will be
      [('train', 0, 60), ('dev', 60, 80), ('test', 80, 100)].
    """
    if len(split_probs) > n_items:
        raise ValueError('Not enough items for the splits. There are {splits} '
                         'splits while there are only {items} items'.format(splits=len(split_probs), items=n_items))
    total_probs = sum(p for name, p in split_probs)
    if abs(1 - total_probs) > 1E-8:
        raise ValueError('Probs should sum up to 1. probs={}'.format(split_probs))
    split_boundaries = []
    sum_p = 0.0
    for name, p in split_probs:
        prev = sum_p
        sum_p += p
        split_boundaries.append((name, int(prev * n_items), int(sum_p * n_items)))

    # Guard against rounding errors.
    split_boundaries[-1] = (split_boundaries[-1][0], split_boundaries[-1][1],
                            n_items)

    return split_boundaries


def _get_inter_splits_by_group(items_and_groups, split_probs, split_number):
    """Split items to train/dev/test, so all items in group go into same split.
    Each group contains all the samples from the same speaker ID. The samples are
    splitted so that all each speaker belongs to exactly one split.
    Args:
      items_and_groups: Sequence of (item_id, group_id) pairs.
      split_probs: List of (split_name, prob), e.g. [('train', 0.6), ('dev', 0.2),
        ('test', 0.2)]
      split_number: Generated splits should change with split_number.
    Returns:
      Dictionary that looks like {split name -> set(ids)}.
    """
    groups = sorted(set(group_id for item_id, group_id in items_and_groups))
    rng = np.random.RandomState(split_number)
    rng.shuffle(groups)

    split_boundaries = _compute_split_boundaries(split_probs, len(groups))

    group_id_to_split = {}
    for split_name, i_start, i_end in split_boundaries:
        for i in range(i_start, i_end):
            group_id_to_split[groups[i]] = split_name

    split_to_ids = collections.defaultdict(set)
    for item_id, group_id in items_and_groups:
        split = group_id_to_split[group_id]
        split_to_ids[split].add(item_id)

    return split_to_ids

def parse_txt_file(txt):
    with open(txt, 'r') as f:
        lines = f.readlines()

    lines = np.array([line.strip().split('\t') for line in lines if line[0] == '['])

    _, fnames, labels, VAD = np.split(lines, lines.shape[1], axis=1)

    VAD = np.array(list(map(ast.literal_eval, VAD.flatten())))  # str to list

    vals, acts, _ = np.split(VAD, 3, axis=1)  # valence-arousal (+ dominance)

    return fnames.flatten(), labels.flatten(), vals.flatten(), acts.flatten()

def filter_label(label):
    if label == 'exc':
        return 'hap'
    elif label in ['xxx', 'fru', 'sur', 'oth', 'fea', 'dis']:
        return None
    else:
        return label

def clip_audio(wrd_path, fs=_SAMPLE_RATE):
    """
    Clip start and end silence according to forced alignment.
    """
    with open(wrd_path, 'r') as f:
        lines = f.readlines()
        lines = np.array([line.strip().split() for line in lines[1:-1]])
        times, words = lines[:, :2].astype(np.int32), lines[:, -1]
        sil_id = (words == '<s>') + (words == '</s>') + (words == '<sil>')
        times, words = times[~sil_id], words[~sil_id]
        start, end = int(fs * times[0, 0] / 100), int(fs * times[-1, -1] / 100)
    return start, end


class Iem4(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for the 4-class IEMOCAP (IEM4) dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    MANUAL_DOWNLOAD_INSTRUCTIONS = """\
    manual_dir should contain the file IEMOCAP_full_release.zip.
    """

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # Specifies the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'audio': tfds.features.Audio(file_format='wav', sample_rate=_SAMPLE_RATE),  # file_format='np.ndarray', dtype=tf.float32, sample_rate=_SAMPLE_RATE),
                'label': tfds.features.ClassLabel(names=set(_LABEL_MAP.values())),
                'arousal': tf.float32,
                'valence': tf.float32,
                'speaker_id': tf.string
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('audio', 'label'),  # Set to `None` to disable
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        self.metadata = {}

        # Downloads the data and defines the splits
        zip_path = os.path.join(dl_manager.manual_dir, 'IEMOCAP_full_release.zip')

        if not tf.io.gfile.exists(zip_path):
            raise AssertionError(
                'IEMOCAP requires manual download of the data. Please download '
                'the audio data and place it into: {}'.format(zip_path))

        extract_path = dl_manager.extract(zip_path)

        txt_paths = sorted(glob(f'{extract_path}/IEMOCAP_full_release/Session*/dialog/EmoEvaluation/*.txt'))
        wav_paths = sorted(glob(f'{extract_path}/IEMOCAP_full_release/Session*/sentences/wav/*'))
        align_paths = sorted(glob(f'{extract_path}/IEMOCAP_full_release/Session*/sentences/ForcedAlignment/*'))

        items_and_groups = []

        for t_path, w_path, a_path in zip(txt_paths, wav_paths, align_paths):
            fnames, labels, vals, acts = parse_txt_file(t_path)

            for i, fname in enumerate(fnames):
                filtered_label = filter_label(labels[i])

                if filtered_label is not None:
                    path = os.path.join(w_path, (fname + '.wav'))
                    audio, fs = sf.read(path)
                    speaker_id = parse_name(fname, from_i=3, to_i=6)
                    items_and_groups.append((path, speaker_id))

                    self.metadata[path] = {
                        'label': filtered_label,
                        'act': acts[i],
                        'val': vals[i]
                    }

        split_probs = [('train', 0.6), ('validation', 0.2), ('test', 0.2)]  # Like SAVEE (https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/audio/savee.py)

        splits = _get_inter_splits_by_group(items_and_groups, split_probs, 0)

        # Returns the Dict[split names, Iterator[Key, Example]]
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={'file_names': splits['train']},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={'file_names': splits['validation']},
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={'file_names': splits['test']},
            ),
        ]

    def _generate_examples(self, file_names):
        for fname in file_names:
            wavname = os.path.basename(fname)
            label = _LABEL_MAP[self.metadata[fname]['label']]
            arousal, valence = self.metadata[fname]['act'], self.metadata[fname]['val']
            speaker_id = parse_name(wavname, from_i=3, to_i=6)
            example = {'audio': fname, 'label': label, 'arousal': arousal, 'valence': valence, 'speaker_id': speaker_id}
            yield fname, example
