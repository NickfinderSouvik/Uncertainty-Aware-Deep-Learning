# When Should AI Say "I Don't Know"?

### Uncertainty-Aware Deep Learning Under Image Corruption

An experimental study of predictive uncertainty, calibration, error detection, and selective prediction in deep learning using the CIFAR-10 image classification dataset.

## Research Question

How reliably can deep learning models estimate their own uncertainty under image corruption, and can uncertainty-calibration techniques improve their ability to identify incorrect predictions?

## Project Overview

Deep learning models can make incorrect predictions with high confidence. This project investigates whether uncertainty estimation can help identify unreliable predictions and allow a model to abstain when it is uncertain.

The study compares a baseline Convolutional Neural Network (CNN), Monte Carlo Dropout, Temperature Scaling, and Deep Ensembles.

## Methods Investigated

- **Baseline CNN:** Reference image classification model.
- **Temperature Scaling:** Post-hoc probability calibration.
- **Monte Carlo Dropout:** Stochastic inference for uncertainty estimation.
- **Deep Ensembles:** Aggregation of predictions from independently trained models.
- **Error Detection:** Evaluating confidence and uncertainty as signals for identifying incorrect predictions.
- **Selective Prediction:** Measuring the trade-off between prediction coverage and classification risk.

## Dataset

**CIFAR-10**

- 60,000 color images
- 10 image classes
- 50,000 training images
- 10,000 test images

The training data were divided into 45,000 training images and 5,000 validation images.

## Experimental Setup

- Framework: PyTorch
- Dataset: CIFAR-10
- Baseline architecture: CNN
- Ensemble size: 3 models
- Training epochs: 10 per model
- Batch size: 128
- Optimizer: Adam
- Learning rate: 0.001
- Hardware used: NVIDIA Tesla T4 GPU

Gaussian noise was applied at different intensity levels to investigate robustness and uncertainty under image corruption.

## Key Validation Findings

- The deep ensemble achieved 73.16% clean validation accuracy compared with 69.46% for the baseline CNN.
- The ensemble improved validation NLL and Brier score, although its clean-validation ECE was higher.
- At 50% coverage, ensemble confidence reduced selective risk by approximately 28.88% compared with the baseline under clean conditions.
- The relative selective-risk reduction generally decreased as corruption severity increased.
- Predictive entropy performed best among the evaluated ensemble uncertainty measures in normalized partial AURC at the two highest noise levels.

These are results from the validation experiments, not a comprehensive independent test-set evaluation.

## Repository Structure

```text
Uncertainty-Aware-Deep-Learning/
│
├── README.md
├── notebooks/
├── results/
├── figures/
└── paper/
```

## Results and Visualizations

The `results/` directory contains experimental CSV files.

The `figures/` directory contains:
- Selective prediction AURC comparison
- Selective risk-coverage curves

## Research Paper

The `paper/` directory is reserved for the research manuscript, including methodology, experimental results, discussion, limitations, conclusion, and IEEE-style references.

## Limitations

- Most comparative experiments were conducted on the validation set.
- Repeated corruption experiments used five noise realizations with fixed trained model parameters.
- The repeated experiments do not measure variability from independently retrained models.
- The corruption study focuses on additive Gaussian noise.
- A comprehensive independent test-set evaluation remains future work.

## Future Work

- Independent test-set evaluation
- Training with multiple random initialization seeds
- Evaluation under additional image corruptions
- Further investigation of calibration and uncertainty estimation
- Evaluation on larger and more diverse datasets

## Technologies

Python | PyTorch | Torchvision | NumPy | Scikit-learn | Matplotlib | Google Colab

## Project Status

Exploratory research project — experimental results and documentation are being organized. Further validation and independent test-set evaluation are planned.

## Author

**NickfinderSouvik**

