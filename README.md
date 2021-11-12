# SERAB: Speech Emotion Recognition Adaptation Benchmark

This repo contains a "simplified" implementation of [SERAB](https://arxiv.org/abs/2110.03414), which includes:
* BYOL-A training and utility functions (Original repo: https://github.com/nttcslab/byol-a)
* BYOL-A and transformer-inspired models
    * Kudos to Phil Wang for his implementation of CvT (https://github.com/lucidrains/vit-pytorch)
* Benchmark tests for SERAB
* TFDS scripts to load SERAB data

## Demo
* A quick demo detailing the SERAB evaluation procedure on a Colab notebook is available [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1EiHujFVt9Hb0VbI0b5RaMfYOYaHq9NrQ?usp=sharing)

## Environment setup
Libraries to reproduce the environment are detailed in `serab.yml`.

To reproduce the environment, run:

```console
conda env create -f serab.yml
```

## Evaluate a (pre-trained model) using SERAB
In this simplified version, only PyTorch models can be used.

Before running the evaluation, make sure that the config file `config.yaml` is correctly setup for your model.

To run a pre-existing model, run:
```console
python clf_benchmark.py --model_name {MODEL_NAME} --dataset_name {DATASET_NAME}
```

By default, grid-search-based classifier hyperparameter optimization is performed. To run a pre-existing model with the "default" classifiers, add the `model_selection --none` key:
```console
python clf_benchmark.py --model_name {MODEL_NAME} --dataset_name {DATASET_NAME} --model_selection none
```

To run a model on all the SERAB datasets, <a href="https://dvc.org/">DVC</a> can be used.

Make the appropriate modifications in `dvc.yaml` and run:
```console
dvc repro
```

## Train a model "à la BYOL-A"
Models can be pre-trained on a subsample of AudioSet that only contains speech.

You might need to do changes in `train.py` and `config.yaml` before starting training.

To train a model, run:
```console
python train.py {MODEL_NAME}  # or dvc repro
```

As training time is usually long (10-20h depending on the model), we recommend using [tmux](https://github.com/tmux/tmux) to attach & detach terminals from a given session.

## SERAB datasets
While CREMA-D and SAVEE are already integrated into TFDS, the other datasets were added as <a href="https://www.tensorflow.org/datasets/add_dataset">tensorflow datasets</a>.

The code to load these datasets can be found in `tensorflow_datasets`.

Here are the steps to download and load the SERAB datasets:
1. In the `tensorflow_datasets` folder, create the folders `download/manual`
2. Download the compressed datasets (.zip files) under `tensorflow_datasets/download/manual/`

Link to the SERAB Datasets:
* AESDD: http://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/
* CaFE: https://zenodo.org/record/1478765
* EmoDB: http://emodb.bilderbar.info/download/
* EMOVO: http://voice.fub.it/activities/corpora/emovo/index.html
* IEM4 (restricted access): https://sail.usc.edu/iemocap/
* RAVDESS: https://smartlaboratory.org/ravdess/
* SAVEE (restricted access): http://kahlan.eps.surrey.ac.uk/savee/Download.html
* ShEMO: https://github.com/mansourehk/ShEMO

3. Build each dataset using the TFDS CLI:
```console
cd tensorflow_datasets/{DATASET_NAME}
tfds build  # Download and prepare the dataset to `~/tensorflow_datasets/
```

The datasets are now ready to use!

## Citation

If you are using this code, please cite [the paper](https://arxiv.org/abs/2110.03414):
```
@article{scheidwasser2021serab,
  title={SERAB: A multi-lingual benchmark for speech emotion recognition},
  author={Scheidwasser-Clow, Neil and Kegler, Mikolaj and Beckmann, Pierre and Cernak, Milos},
  journal={arXiv preprint arXiv:2110.03414},
  year={2021}
}
```
