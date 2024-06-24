# Unsupervised Transformer Model for Time Series Data

This code includes a transformer-based framework for unsupervised representation learning of multivariate time series, inspired by [Zerveas et al.](https://dl.acm.org/doi/10.1145/3447548.3467401).

The model is trained using the missing value imputation task to create embeddings that potentially extract complex features from pedestrian trajectories. 
These embeddings are subsequently used for clustering to reveal different behaviors. These behavior clusters are combined with data-driven reachability analysis, yielding an end-to-end data-driven approach to predicting the future motion of pedestrians.


## Setup
```bash
cd Pedestrian_Prediction/
pip install -r requirements.txt
```

The following commands assume that you have created a new root directory inside the project directory like this: 
`mkdir experiments`. Inside this already *existing* root directory, each experiment will create a time-stamped output directory containing
model checkpoints, performance metrics per epoch, the experiment configuration, log files, etc. 

*Python >= 3.9*

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

To see all command options with explanations, run: `python main.py --help`

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
