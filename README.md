# Naive Bayes Abstract Classifier

A Multinomial Naive Bayes text classifier built from scratch in Python, trained to classify 
biological research abstracts into four domains: Archaea, Bacteria, Eukaryota, and Virus.

## Results

| Laplace Value | Training Accuracy | Test Accuracy |
|---------------|-------------------|---------------|
| 0.3           | 99.58%            | 96.6%         |
| 0.5           | 99.38%            | 98.0%         |
| 0.7           | 98.88%            | 97.0%         |
| 1.0           | 97.95%            | 96.3%         |
| 1.5           | 96.45%            | too low       |

Best result: **98.0% test accuracy** at Laplace smoothing value of 0.5

## How it works

- **Bag of words** — each abstract is represented as word frequency counts across four classes
- **Log-space probabilities** — avoids numerical underflow from multiplying many small numbers
- **Laplace smoothing** — prevents zero probability for unseen words
- **Custom preprocessing** — lowercasing, stopword removal, and merging of biological 
  multi-word terms (e.g. "escherichia coli" → "escherichia-coli")

## Setup

Install dependencies:
```bash
pip install numpy pandas jupyter
```

Open the notebook:
```bash
jupyter notebook naive_bayes.ipynb
```

## Files

- `naive_bayes.ipynb` — full implementation and report
- `train.csv` — training abstracts
- `test.csv` — test abstracts

## What I learned

Implemented Naive Bayes entirely from scratch without sklearn, gaining a solid understanding 
of how text classifiers work under the hood — including why log-space computation matters 
and how smoothing affects the bias-variance tradeoff.
