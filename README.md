# NASTAR: NOISE ADAPTIVE SPEECH ENHANCEMENT WITH TARGET-CONDITIONALRESAMPLING
This is the official implementation of our [paper](https://changlee0903.github.io/NASTAR-demo). 
Our work applies resampling techniques on existing source datasets to the noise adaptive speech enhancement task in a one-shot scenario. The resampling scheme can achieve higher scores with less training data.

## Current Information
This project is partially public. The source code of SE has been released. We will open the implementation of our resampling method if our paper gets accepted. 
Those [checkpoints and logs](https://drive.google.com/drive/folders/1BQ5Y9618OADVC7mbhfkqTLkqYf0g4Rt9?usp=sharing) under all the noise adaptation settings and the [testing data](https://drive.google.com/file/d/1HnWjKwSIRN4ieI19oifzh1-GDEiWw32q/view?usp=sharing) are provided.
The pseudo-noise and relevant-cohort demonstration for NASTAR can be found [here](https://changlee0903.github.io/NASTAR-demo/).

## Contents
- [Installation](#installation)
- [Steps and Usages](#steps-and-usages)
- [Contact](#contact)

## Installation
Note that our environment is in ```Python 3.7.10```. To run the experiment of NASTAR, you can clone the repository and install it by using the pip tool:
```bash
git clone https://github.com/ChangLee0903/NASTAR
cd NASTAR
# Install all the necessary packages
pip install -r requirements.txt
```

## Steps and Usages
1. Data Preprocess:

   You can produce your own pair data by using the <code>Corruptor</code> class in <code>data.py</code> and set the data path in <code>config/config.yaml</code>. 
Note that all the training noisy utterances are generated online. Our source noise dataset is provided by [DNS-Challenge](https://github.com/microsoft/DNS-Challenge). 
The <code>target_data</code> directory contains the noise signals <code>test.wav</code> used for testing and the results of pseudo-noise <code>pesudo.wav</code> / cohort-set <code>cohort.txt</code>.
Check <code>dataset</code> in <code>config/config.yaml</code>:
   <pre><code>dataset:
    train:
      speech: ../speech_data/LibriSpeech/train-clean-360
      noise: ../noise_data/DNS_noise
    dev:
      speech: ../speech_data/LibriSpeech/dev-clean
    test:
      data: ../NASTAR_VCB_test_data
    ...
    </code></pre>
   Since we only test adapted model on one specifice noise type, there is no need to set the noise dataset path in <code>dev</code>. 
Instead, we use <code>--eval_noise</code> to assign the noise signal for evaluation. 
To avoid randomness as testing, the testing data have been mixed and put in <code>NASTAR_VCB_test_data</code>. All the noisy and clean utterances were saved as <code>npy</code> files.

2. Training the adapted SE model for each target noise condition:

   First, download the checkpoint of pretrained model in the <code>PTN</code> directory, and make sure the argument <code>--ae_ckpt</code> has been set correctly.
   All the recipes of different settings are recorded in our repository so that you can just run the script. 
   Note that you can set <code>--device</code> to change the identity of the used CUDA device.
   <pre><code>bash train_NASTAR.sh
   </code></pre>


3. Evaluation:

    Make sure all of your models have been trained and stored in the path of <code>--ckptdir</code>. All the results will be stored in dictionaries and saved as <code>pth</code> files.
    - Check ckpt directory
    - Run:

   <pre><code>python main.py --task test --n_jobs 16</code></pre>
    
    
## Contact
Any bug report or improvement suggestion will be appreciated!
```bash
e-mail: r08922a28@csie.ntu.edu.tw
```
