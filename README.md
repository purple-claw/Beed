# BEED Research Notebook

I use this project to study the BEED dataset, short for Bangalore EEG Epilepsy Dataset, through an exploratory notebook and a second notebook focused on model training and PEFT-style fine-tuning. The goal is not only to build a classifier, but to understand the full process end to end: the dataset structure, the preprocessing decisions, the baseline model, and the effect of a low-rank adaptation step.

## What I Am Working On

I keep this project in two parts:

1. I inspect the data visually and statistically in `datVis.ipynb`.
2. I build and compare training phases in `mnomMdl.ipynb`.

The notebooks are written as a research log, so I keep the analysis readable and the implementation simple enough to explain line by line.

## BEED Dataset

I work with `BEED_Data.csv`, which contains:

- 16 numeric feature columns: `X1` through `X16`
- 1 target column: `y`
- 8000 total rows
- 4 balanced classes in the target

The dataset is multivariate, integer-valued, and arranged as a CSV classification problem. It represents EEG-derived measurements collected from 80 individuals, with 20 subjects per folder across four folders and 200 time points per subject.

### Dataset Details I Use

- Subject area: Health and Medicine
- Instances: 8,000, with 2,000 samples per class
- Features: 16 EEG channels
- Task type: Multi-class classification
- Missing values: none reported
- Signal basis: EEG measurements sampled over 20-second intervals at 256 Hz

### Target Variable

I treat `y` as a multi-class target with four labels:

- `0`: healthy subjects, control group without epileptic seizures
- `1`: generalized seizures
- `2`: focal seizures
- `3`: seizure events, including activities such as eye blinking or constant staring

I also note that the dataset can be reframed for binary seizure detection by collapsing the labels into seizure vs. no seizure, but in this project I keep the full 4-class formulation so I can study the finer-grained diagnosis problem.

### EEG Channel Structure

I interpret the 16 features as EEG channels mapped to brain regions under the international 10-20 electrode placement system:

- Frontal lobe channels for frontal brain activity
- Temporal lobe channels for temporal brain activity
- Parietal lobe channels for parietal brain activity
- Occipital lobe channels for occipital brain activity
- Central channels for central brain regions

These channels give me a compact representation of the EEG signal characteristics that I can use for classification.

From the data inspection, I treat this as a multiclass tabular classification problem. The classes are evenly distributed, so I do not need class reweighting for the current phase.

### What I Learned About the Data

I observe that the feature values are centered around negative and positive numeric ranges with similar scale across columns. I also see strong correlations between several neighboring features, which suggests the dataset has internal structure instead of being random noise.

The exploratory notebook focuses on:

- reading the CSV
- checking the shape, dtypes, and descriptive statistics
- plotting feature distributions
- measuring skewness
- visualizing the correlation matrix
- identifying highly and moderately correlated feature pairs

## Project Structure

- `BEED_Data.csv` holds the dataset.
- `datVis.ipynb` contains the exploratory data analysis workflow.
- `mnomMdl.ipynb` contains the model training workflow with the baseline and PEFT-style phase.

## Research Workflow

I follow a simple research-style pipeline:

1. I inspect the dataset and confirm its structure.
2. I normalize the numeric features using only the training split statistics.
3. I train a weak baseline model to establish a reference point.
4. I train a stronger model and then fine-tune it with a LoRA-style low-rank adapter approach.
5. I compare the final accuracies so I can see whether the parameter-efficient step helps.

## Phase 1: Baseline Model

I start with a deliberately weak baseline so the improvement from the later phase is easy to see. In `mnomMdl.ipynb`, I:

- split the data into train, validation, and test sets using a stratified class-wise split
- standardize the features with training-set mean and standard deviation
- keep only the first 8 features for the weak baseline
- train a small softmax classifier with gradient descent
- track validation accuracy and keep the best checkpoint

This baseline is intentionally limited. I use it as a learning reference point rather than as the final model.

## Phase 1: Improved Model With LoRA and PEFT

After the weak baseline, I move to a low-rank adaptation workflow. I use a slightly deeper NumPy model as a backbone, then I freeze the backbone weights and train small low-rank adapter matrices on top of them.

What I do in this phase:

- train a backbone model on the full feature space
- freeze the backbone after pretraining
- add low-rank adapter parameters to the linear layers
- optimize only the adapter parameters during fine-tuning
- compare the final test accuracy against the baseline

This gives me a practical PEFT demonstration without depending on a heavy deep learning stack.

## Intended Use Cases

I frame the project around a few common medical AI tasks:

- binary seizure detection for healthy vs. seizure-present classification
- multi-class seizure identification for labels `0`, `1`, `2`, and `3`
- time-series style signal analysis and pattern recognition
- feature engineering for statistical and frequency-domain exploration
- deep learning prototyping for CNN, LSTM, or hybrid approaches in future phases
- real-time seizure alert system design as a longer-term research direction

## Problem Statement

I treat epilepsy detection as a practical medical AI problem because manual EEG review is time-consuming, labor-intensive, and difficult to scale for real-time monitoring. This project explores whether a compact machine learning workflow can support automated seizure detection and classification with clearer, faster decision support.

### Why I Use This Setup

I choose this path because it shows the idea behind PEFT clearly:

- the baseline shows what happens when capacity is limited
- the pretrained backbone shows what happens when the full feature space is used
- the LoRA-style step shows how I can adapt a model efficiently with a small number of trainable parameters

## Notebook Notes

I keep the notebook implementation self-contained with NumPy, pandas, and matplotlib so it runs in a lightweight environment. I also avoid unnecessary package dependencies where I can.

If I run the notebooks in order, I get a full workflow from exploration to comparison.

## How I Run It

I open the notebooks in this order:

1. `datVis.ipynb` for data inspection and visualization.
2. `mnomMdl.ipynb` for the training pipeline.

In the training notebook, I run the cells from top to bottom so the dataset, helper functions, splits, baseline, and PEFT phase are defined in the right sequence.

## What I Expect To See

I expect the baseline to perform noticeably worse than the improved PEFT phase. That contrast is the main educational goal of the project. I use the comparison to understand how much value I get from:

- using the full feature set
- training a backbone first
- fine-tuning with low-rank adapters instead of retraining everything

## Status

I have the exploratory notebook and the two training phases in place. The next phases can build on this foundation without changing the core data pipeline.
