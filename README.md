# ml-notebook-vault

> A personal study vault of 50+ machine learning implementations — from classical algorithms to deep learning, NLP, computer vision, and LLM fine-tuning. Built over 1.5+ years of hands-on learning using Python, scikit-learn, PyTorch, and HuggingFace.

---

## What's Inside

This repository documents my ML learning journey from the ground up. Every notebook is an implementation — not a tutorial copy — written to understand the math, the code, and the intuition behind each algorithm.

---

## Contents

### Classical Machine Learning

| Notebook | Algorithm | Notes |
|---|---|---|
| `linear_regression` | Simple & Multiple Linear Regression | Salary prediction, polynomial extension |
| `logistic_regression` | Logistic Regression | Binary classification |
| `decision_tree_classification` | Decision Tree | Entropy / Gini split |
| `decision_tree_regression` | Decision Tree Regressor | Continuous output |
| `random_forest_regression` | Random Forest | 0.91 accuracy on Social Ads dataset |
| `support_vector_machine` | SVM (Kernel) | RBF, linear, polynomial kernels |
| `support_vector_regression` | SVR | Regression with margin tolerance |
| `k_nearest_neighbors` | KNN | Distance-based classification |
| `naive_bayes` | Naive Bayes | Probabilistic text/tabular classification |
| `xg_boost` | XGBoost | Gradient boosting ensemble |
| `upper_confidence_bound` | UCB (Reinforcement) | Multi-armed bandit / ad selection |

### Model Selection & Optimization

| Notebook | Topic |
|---|---|
| `k_fold_cross_validation` | K-Fold CV — generalization evaluation |
| `grid_search` | Hyperparameter tuning via GridSearchCV |
| `polynomial_regression` | Feature engineering for non-linear data |

### Dimensionality Reduction

| Notebook | Algorithm |
|---|---|
| `principal_component_analysis` | PCA — unsupervised dimensionality reduction |
| `kernel_pca` | Kernel PCA — non-linear PCA |
| `linear_discriminant_analysis` | LDA — supervised dimensionality reduction |

### Clustering

| Notebook | Algorithm |
|---|---|
| `k_means_clustering` | K-Means — centroid-based clustering |
| `hierarchical_clustering` | Agglomerative clustering with dendrograms |

### Association Rule Learning

| Notebook | Algorithm |
|---|---|
| `apriori` | Apriori — market basket analysis |
| `eclat` | ECLAT — transaction-based frequent itemsets |

### Deep Learning & Neural Networks

| Notebook | Topic |
|---|---|
| `artificial_neural_network` | ANN from scratch — forward/back propagation |
| `convolutional_neural_network` | CNN — image classification |
| `cnn_earlystopping` | CNN with early stopping callback |
| `cifar_cnn` | CIFAR-10 classification with CNN |
| `scratch_cnn` | CNN implemented without high-level wrappers |
| `digit_recognition` | Handwritten digit recognition (MNIST) |
| `pc_lstm` | LSTM — sequence modeling |

### Computer Vision

| Notebook | Topic |
|---|---|
| `plant_disease_cnn` | CNN for plant disease detection from leaf images |
| `deepgrns` | Deep learning applied to gene regulatory networks |

### Natural Language Processing

| Notebook | Topic |
|---|---|
| `natural_language_processing` | Text preprocessing, BoW, TF-IDF |
| `nlp` | NLP classification pipeline |
| `machine_translation` | Seq2Seq / translation model |
| `emotion_detector_model` | Emotion classification from text |
| `bs4_scraping` | Web scraping with BeautifulSoup for NLP data |

### Embeddings & Semantic Search

| Notebook | Topic |
|---|---|
| `embeddings_generator` | Generating sentence embeddings with transformers |
| `embeddings_generator2` | Extended embedding pipeline with chunking |

### LLM Fine-Tuning

| Notebook | Topic |
|---|---|
| `fine_tuning_lora` | LoRA fine-tuning on instruction datasets |
| `searcher_lora_fine_tuning` | Domain-specific LoRA fine-tuning experiment |
| `15_small` | Small model fine-tuning experiment |
| `train_classifier_agent` | Fine-tuned classifier agent pipeline |

### Medical AI

| Notebook | Topic |
|---|---|
| `medical_symptoms` | Symptom-based disease prediction (supports Doctor AI project) |

---

## Tech Stack

- **Languages:** Python
- **ML:** scikit-learn, XGBoost, NumPy, Pandas, Matplotlib, Seaborn
- **Deep Learning:** PyTorch, TensorFlow/Keras
- **NLP & LLMs:** HuggingFace Transformers, NLTK, spaCy, PEFT (LoRA)
- **Computer Vision:** OpenCV, PIL, torchvision
- **Data Collection:** BeautifulSoup4, Requests
- **Environment:** Google Colab, Jupyter Notebook

---

## Note on Notebook Status

Some notebooks are complete and tested end-to-end. Others are experimental or mid-progress. All contain working code blocks — they represent active learning, not polished production code.

---

## Related Projects

These notebooks form the foundation behind production projects including:

- [Doctor AI](https://github.com/sarimahsan101) — Medical symptom checker using NLP + CNN
- [ArXiv Lens](https://github.com/sarimahsan101) — RAG research tool built on transformer embeddings
- [FlowForge](https://github.com/sarimahsan101) — Multi-agent LLM system (AI Agent Olympics, Milan 2026)
- [HuggingFace Models](https://huggingface.co/sarimahsan101) — 3 fine-tuned LLMs published

---

## Author

**Syed Muhammad Sarim Ahsan**
[GitHub](https://github.com/sarimahsan101) · [HuggingFace](https://huggingface.co/sarimahsan101) · [LinkedIn](https://linkedin.com/in/sarimahsan) · [sarimahsan.dev](https://sarimahsan.dev)
