import os

from pathlib import Path

import librosa
import numpy as np
import opensmile
import pandas as pd
import torch
import tensorflow_datasets as tfds
import tensorflow_hub as hub

from pydub import AudioSegment
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from byol_a.common import load_yaml_config
from byol_a.models.audio_ntt import AudioNTT2020
from byol_a.models.cvt import CvT
from settings import REQUIRED_SAMPLE_RATE


def compute_norm_stats(dataset_name, audios, orig_sr, to_melspec):
    """Compute dataset mean and std for pre-normalization.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset
    audios : list
        List of audio samples
    orig_sr : int
        Sample rate of the raw audios

    Returns
    -------
    stats : list
        [mean, std] of the dataset
    """
    mean = 0.
    std = 0.

    for int_audio in tqdm(audios, desc=f'Computing stats for {dataset_name}', total=len(audios), ascii=True):
        # Convert to float
        if int_audio.max() > np.iinfo(np.int16).max:
            float_audio = np.float32(int_audio.astype(np.float32) / np.iinfo(np.int32).max)
        else:
            float_audio = int_audio.astype(np.float32) / np.iinfo(np.int16).max

        # Resample if needed
        if orig_sr != REQUIRED_SAMPLE_RATE:
            float_audio = librosa.core.resample(
                float_audio,
                orig_sr=orig_sr,
                target_sr=REQUIRED_SAMPLE_RATE,
                res_type='kaiser_best'
            )

        # Convert to tensor
        float_audio = torch.tensor(float_audio).unsqueeze(0)

        # Compute log-mel-spectrogram
        lms = (to_melspec(float_audio) + torch.finfo(torch.float).eps).log()

        # Compute mean, std
        mean += lms.mean()
        std += lms.std()

    mean /= len(audios)
    std /= len(audios)

    stats = [mean.item(), std.item()]

    print(f'Finished {dataset_name}')

    return stats


def load_model(model_name, cfg, device, ckpt_folder='checkpoints/'):
    """Load pre-trained DL models.

    Parameters
    ----------
    model_name : str
        Model name
    cfg : EasyDict
        Configuration details from config.yaml
    device : torch.device
        Used device (CPU or GPU)
    ckpt_folder : str, Optional
        Path to checkpoint folder, by default checkpoints/

    Returns
    -------
    torch.nn.Module or a tensorflow "trackable" object
        Model loaded with pre-training weights

    Raises
    ------
    ValueError
        If the model configuration is erroneous
        N.B. A PyTorch model should be loaded with pre-trained weights
    """
    weight_file = None

    # Load pretrained weights.
    if model_name == 'default':
        model = AudioNTT2020(n_mels=cfg.n_mels, d=cfg.feature_d)

        if cfg.feature_d == 2048:
            weight_file = 'default2048_BYOLAs64x96-2105311814-e100-bs256-lr0003-rs42.pth'  # Retrained model
            # weight_file = 'AudioNTT2020-BYOLA-64x96d2048.pth'  # Pretrained model (original version)
        elif cfg.feature_d == 1024:
            weight_file = 'default1024_BYOLAs64x96-2107292000-e100-bs256-lr0003-rs42.pth'  # Retrained model
            # weight_file = 'AudioNTT2020-BYOLA-64x96d1024.pth'  # Pretrained model (original version)
        elif cfg.feature_d == 512:
            weight_file = 'default512_BYOLAs64x96-2107281850-e100-bs256-lr0003-rs42.pth'  # Retrained model
            # weight_file = 'AudioNTT2020-BYOLA-64x96d512.pth'  # Pretrained model (original version)
        else:
            raise ValueError(f'{model_name} config not found for embedding dimension = {cfg.feature_d}. Try feature_d = 512, 1024 or 2048.')

        weight_file = Path(ckpt_folder, weight_file)

    elif model_name == 'cvt':
        s1_depth, s2_depth, s3_depth = cfg.depths
        s1_emb_dim, s2_emb_dim, s3_emb_dim = cfg.embed_dims
        s1_mlp_mult, s2_mlp_mult, s3_mlp_mult = cfg.mlp_mults

        model = CvT(
            s1_emb_dim=s1_emb_dim,
            s1_depth=s1_depth,
            s1_mlp_mult=s1_mlp_mult,
            s2_emb_dim=s2_emb_dim,
            s2_depth=s2_depth,
            s2_mlp_mult=s2_mlp_mult,
            s3_emb_dim=s3_emb_dim,
            s3_depth=s3_depth,
            s3_mlp_mult=s3_mlp_mult,
            pool=cfg.cvt_pool
        )

        if (s1_depth == 1) & (s2_depth == 1) & (s3_depth == 1):
            if (s2_emb_dim == 128) & (s3_emb_dim == 256):
                # CvT 1-1-1 (64-128-256)
                weight_file = 'cvt_s1-d1_s2-d1_s3-d1_BYOLAs64x96-2106251634-e100-bs512-lr0003-rs42.pth'
            elif (s2_emb_dim == 256) & (s3_emb_dim == 512):
                if cfg.cvt_pool == 'mean+max':
                    # CvT 1-1-1 (64-256-2048)
                    # Same as 64-256-512, but with mean+max temporal aggregation --> embedding size = 2048
                    weight_file = 'cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-2107301623-e100-bs256-lr0003-rs42.pth'
                else:
                    # CvT 1-1-1 (64-256-512)
                    weight_file = 'cvt_s1-d1-e64_s2-d1-e256_s3-d1-e512_BYOLAs64x96-2107021643-e100-bs512-lr0003-rs42.pth'
            else:
                raise ValueError(f'{model_name} config not found for CvT depths = {(s1_depth, s2_depth, s3_depth)}.')
        else:
            raise ValueError(f'{model_name} config not found.')

        weight_file = Path(ckpt_folder, weight_file)

    elif model_name == 'vggish':
        model = hub.load('https://tfhub.dev/google/vggish/1')
    elif model_name == 'yamnet':
        model = hub.load('https://tfhub.dev/google/yamnet/1')
    elif model_name == 'trill':
        model = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3')
    elif model_name == 'opensmile':
        model = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    else:
        raise ValueError('Model not found.')

    if weight_file is not None:
        state_dict = torch.load(weight_file, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        # Disable parameter tuning
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return model, weight_file


def generate_embeddings(
    model,
    model_name,
    audios,
    split,
    orig_sr,
    to_melspec,
    normalizer,
    device
):
    """
    Generate audio embeddings from a pretrained feature extractor.

    Converts audios to float, resamples them to the desired learning_rate,
    and produces the embeddings from a pre-trained model.

    Adapted from https://github.com/google-research/google-research/tree/master/non_semantic_speech_benchmark

    Parameters
    ----------
    model : torch.nn.Module object or a tensorflow "trackable" object
        Model loaded with pre-training weights
    model_name : str
        Name of the model
    audios : list
        List of audios, loaded as a numpy arrays
    split : str
        Dataset split, can be 'train', 'validation' or 'test'.
    orig_sr : int
        Original sample rate in the dataset.
    to_melspec : torchaudio.transforms.MelSpectrogram object
        Mel-spectrogram transform to create a spectrogram from an audio signal
    normalizer : nn.Module
        Pre-normalization transform
    device : torch.device object
        Used device (CPU or GPU)

    Returns
    -------
    embeddings: numpy.ndarray
        2D Array of embeddings for each audio of size (N, M). N = number of samples, M = embedding dimension
    """
    embeddings = []
    for int_audio in tqdm(audios, desc=f'Generating embeddings for {split}', total=len(audios), ascii=True):
        # Convert to float
        if int_audio.max() > np.iinfo(np.int16).max:
            float_audio = np.float32(int_audio.astype(np.float32) / np.iinfo(np.int32).max)
        else:
            float_audio = int_audio.astype(np.float32) / np.iinfo(np.int16).max

        if orig_sr != REQUIRED_SAMPLE_RATE:
            float_audio = librosa.core.resample(
                float_audio,
                orig_sr=orig_sr,
                target_sr=REQUIRED_SAMPLE_RATE,
                res_type='kaiser_best'
            )

        if model_name == 'trill':
            embedding = np.mean(model(float_audio, sample_rate=REQUIRED_SAMPLE_RATE)['layer19'], axis=0, keepdims=True)

        elif model_name == 'vggish':
            if len(float_audio) < REQUIRED_SAMPLE_RATE:
                to_pad = REQUIRED_SAMPLE_RATE - len(float_audio)
                pre_pad = np.random.randint(0, to_pad)
                post_pad = to_pad - pre_pad
                assert pre_pad + post_pad == to_pad

                float_audio = np.pad(float_audio, (pre_pad, post_pad))

            if model(float_audio).shape[0] == 0:
                print(float_audio.shape, float_audio.min(), float_audio.max())
                assert False
            embedding = np.mean(model(float_audio), axis=0, keepdims=True)

        elif model_name == 'yamnet':
            _, embedding, _ = model(float_audio)
            embedding = np.mean(embedding, axis=0, keepdims=True)

        elif model_name == 'opensmile':
            embedding = model.process_signal(float_audio, REQUIRED_SAMPLE_RATE)

        else:
            float_audio = torch.tensor(float_audio).unsqueeze(0)
            lms = normalizer((to_melspec(float_audio) + torch.finfo(torch.float).eps).log()).unsqueeze(0)
            embedding = model(lms.to(device)).cpu().detach().numpy()

        embeddings.append(embedding)

    print(f'Finished {split}')
    embeddings = np.squeeze(np.array(embeddings, dtype=np.float32), 1)
    return embeddings


def speaker_normalization(embedding_np, speaker_id_np):
    """Normalize embedding features with per-speaker statistics.

    Adapted from https://github.com/google-research/google-research/tree/master/non_semantic_speech_benchmark

    Parameters
    ----------
    embedding_np: numpy.ndarray
        2D Array of embeddings for each audio of size (N, M). N = number of samples, M = embedding dimension

    speaker_id_np: numpy.ndarray
        An array of size (N,) with the speaker ID for each audio.

    Returns
    numpy.ndarray
        Speaker-normalized 2D embeddings
    """
    all_speaker_ids = np.unique(speaker_id_np)
    for speaker in all_speaker_ids:
        cur_i = speaker_id_np == speaker
        embedding_np[cur_i] -= embedding_np[cur_i].mean(axis=0)
        stds = embedding_np[cur_i].std(axis=0)
        stds[stds == 0] = 1
        embedding_np[cur_i] /= stds
    return embedding_np


def save_results(fname, results_df, results_folder):
    """Save a DataFrame into a csv and update it if already existing.

    Parameters
    ----------
    fname : str
        CSV filename
    results_df : pandas.DataFrame
        New results to be saved
    results_folder : str
        Folder/path to CSV filename
    """
    save_path = Path(results_folder, fname)

    os.makedirs(results_folder, exist_ok=True)

    if not(os.path.exists(save_path)):
        print(f'File {fname} does not exist yet. Creating a new results file.')
        results_df.to_csv(save_path)
    else:
        all_results_df = pd.read_csv(save_path, index_col=0)
        new_all_results_df = pd.concat([results_df, all_results_df], axis=1)
        new_all_results_df = new_all_results_df.reindex(sorted(new_all_results_df.columns), axis=1)
        new_all_results_df.to_csv(save_path)

def stereo2mono(files):
    """Convert all files from stereo to mono.
    Note: this would effectively also create a copy of files that were already in a mono format

    Inspired by: https://stackoverflow.com/questions/5120555/how-can-i-convert-a-wav-from-stereo-to-mono-in-python
    Parameters
    ----------
    files : iterable
        Sequence of files
    Example use:
    ```
    from glob import glob
    files = glob('path\to\your\audios\*.mp3')
    stereo2mono(files)
    ```
    Then you may remove the files that do not contain the '_mono' tag.
    """
    for f in files:
        # Load audio
        sound = AudioSegment.from_wav(f)
        # Convert to mono
        sound = sound.set_channels(1)
        # Save file
        stem, ext = os.path.splitext(f)
        sound.export(f'{stem}_mono{ext}', format='wav')
