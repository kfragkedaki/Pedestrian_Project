# Unsupervised Transformer Model for Time Series Data

This code includes a transformer-based framework for unsupervised representation learning of multivariate time series, inspired by 
the [paper](https://dl.acm.org/doi/10.1145/3447548.3467401): George Zerveas et al. **A Transformer-based Framework for Multivariate Time Series Representation Learning**, in _Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '21), August 14-18, 2021_.
ArXiV version: https://arxiv.org/abs/2010.02803.

The model is trained using the missing value imputation task to create embeddings that potentially extract features from the time series data. 
These embeddings are later used for additional downstream tasks, such as clustering.


## Setup
```bash
cd Pedestrian_Prediction/
pip install -r requirements.txt
```

The following commands assume that you have created a new root directory inside the project directory like this: 
`mkdir experiments`. Inside this already *existing* root directory, each experiment will create a time-stamped output directory containing
model checkpoints, performance metrics per epoch, the experiment configuration, log files, etc.

### Get SinD Dataset

The data used in this repository correspond to the [paper](https://arxiv.org/abs/2209.02297). 
You can get the data by executing `git clone https://github.com/SOTIF-AVLab/SinD.git`. 
To access the full dataset, you need to contact the authors.

```bash
mkdir resources
cd resources
git clone https://github.com/SOTIF-AVLab/SinD.git
```

## Run

To see all command options with explanations, run: `python src/main.py --help`

### Train
The downstream task of missing value imputation is used to train the model. The data given are split into training and validation data and the loss metric is used for optimization.
The embedding of are saved under the 
```bash
python main.py --output_dir ./experiments --comment "pretraining through imputation" --name SINDDataset_pretrained --data_dir resources/SinD/Data --data_class sind --pattern Ped_smoothed_tracks --pos_encoding learnable --harden
```

### Evaluation
To evaluate the model you can use the *eval_only* parameter. This parameter evaluates all the data given and extracts their embedding into the rood directory */experiments*.

```bash
python main.py --comment "evaluation" --name SINDDataset_evaluation --data_dir resources/SinD/Data --data_class sind --pattern Ped_smoothed_tracks --pos_encoding fixed --eval_only --load_model experiments/SINDDataset_pretrained_$time$/checkpoints/model_best.pth
```

### Output
Besides the console output and the logfile `output.log`, you can monitor the evolution of performance with:
```bash
tensorboard --logdir path/to/output_dir
```
