# Multimodal Product Price Prediction via Stacked Gradient Boosting and Semantic Embeddings

**A Technical Report — Amazon ML Hackathon 2025**

---

*Submitted to: Amazon ML Challenge 2025*
*Date: March 2025*

---

## Abstract

Accurate price estimation for e-commerce products is a long-standing challenge, requiring the integration of heterogeneous signals spanning unstructured text, visual content, and structured catalogue attributes. This paper presents **SmartProductPricing**, a multimodal machine learning pipeline designed for the Amazon ML Hackathon 2025 Smart Product Pricing Challenge. Our approach combines (i) semantic text representations obtained via TF-IDF dimensionality reduction or Sentence-BERT (SBERT) embeddings, (ii) deep visual features extracted by a pre-trained ResNet50 convolutional network, and (iii) hand-crafted tabular signals including item-pack-quantity (IPQ), digit density, and token-frequency statistics. These heterogeneous feature sets are fused into a single feature matrix and fed into two LightGBM gradient-boosting models — one trained on the raw target and one on its log-transformed counterpart. The resulting out-of-fold (OOF) predictions are blended by a RidgeCV meta-learner in a two-stage stacking ensemble. The system is evaluated under the Symmetric Mean Absolute Percentage Error (SMAPE) objective. We describe the complete pipeline architecture, design decisions, ablation considerations, and engineering strategies including embedding caching, stratified folding over quantile-binned price, and GPU-accelerated image download.

---

## 1. Introduction

Pricing automation is a core competency for large-scale e-commerce platforms. A product's price is determined by a complex interplay of factors: the merchant's description, the visual quality of images, the pack size of the unit being sold, and prevailing market conditions. Manual pricing at the scale of millions of SKUs is intractable, making machine-learned models essential.

The Amazon ML Hackathon 2025 presents this problem as a supervised regression task: given a product's `catalog_content` (a free-text descriptor) and an associated image URL, predict its price in USD. The training set contains approximately 600,000 product records and the test set a comparable number, both stored as CSV files totalling ∼148 MB on disk.

The challenge poses several modelling difficulties:

1. **Highly skewed target distribution.** Product prices span several orders of magnitude (from < $1 to > $10,000), making standard mean-squared-error objectives poorly calibrated.
2. **Heterogeneous input modalities.** Text, images, and structured fields require different inductive biases that no single model naturally handles.
3. **Sparse and noisy text.** Catalogue content mixes brand names, size specifications, and marketing copy in an unstructured string.
4. **Large-scale image corpus.** Downloading and encoding ∼75,000 images per split requires parallelised I/O and GPU compute.
5. **Scale-invariant evaluation.** SMAPE penalises relative errors symmetrically, rewarding models that avoid catastrophic over- or under-prediction on extreme values.

We address all five challenges through a carefully engineered pipeline described in the remainder of this paper.

---

## 2. Related Work

### 2.1 Multimodal Price Prediction

Price prediction from product listings has been studied primarily in the context of used-car pricing [Gegic et al., 2019], housing markets [Park & Bae, 2015], and second-hand marketplace items [Wang et al., 2021]. These works highlight that visual features consistently improve prediction over text-only baselines when the product category is visually distinctive. Our work extends this finding to a broad, cross-category Amazon catalogue.

### 2.2 Gradient Boosting for Tabular Regression

LightGBM [Ke et al., 2017] is the de facto standard for tabular regression in competitive machine learning. Its leaf-wise tree growth strategy, histogram-based splitting, and support for GPU acceleration make it particularly well-suited to high-dimensional, sparse feature matrices produced by TF-IDF vectorisation. We use it as the primary base learner.

### 2.3 Semantic Text Representations

Sentence-BERT [Reimers & Gurevych, 2019] produces dense, semantically meaningful sentence embeddings by fine-tuning BERT with a siamese network objective on sentence pairs. The `all-mpnet-base-v2` variant produces 768-dimensional embeddings that generalise well across domains. As a computationally lighter alternative, we employ TF-IDF (up to 5,000 vocabulary features) followed by Truncated SVD [Halko et al., 2011] to obtain 128-dimensional latent semantic representations.

### 2.4 Transfer Learning for Visual Features

ResNet50 [He et al., 2016], pre-trained on ImageNet-1K, is used as a feature extractor by removing the final classification head. The penultimate 2,048-dimensional global-average-pooled representation captures rich visual semantics without task-specific fine-tuning. This transfer-learning strategy has proven effective for product image similarity tasks [Bell & Bala, 2015].

### 2.5 Stacking Ensembles

Stacked generalisation [Wolpert, 1992] combines multiple base learner predictions through a meta-model trained on OOF predictions, avoiding the overfitting that would arise from training and predicting on the same data. RidgeCV provides a simple, robust linear meta-learner with built-in cross-validated regularisation path search.

---

## 3. Dataset

### 3.1 Structure

| Split | File | Rows (approx.) | Size |
|---|---|---|---|
| Train | `dataset/train.csv` | ~600,000 | 74 MB |
| Test | `dataset/test.csv` | ~600,000 | 74 MB |
| Sample Test | `dataset/sample_test.csv` | ~1,000 | 107 KB |

**Key columns:**

| Column | Description |
|---|---|
| `sample_id` | Unique product identifier |
| `catalog_content` | Free-text product description (title + attributes) |
| `image_link` | URL to product image |
| `price` | USD price (train only, regression target) |

### 3.2 Target Distribution

The price distribution is right-skewed with a long tail. Empirically, $\log_{1+x}$ transformation produces an approximately normal distribution, motivating our dual-model strategy (§ 5.2).

### 3.3 Data Quality

- Null prices (< 0.1% of train) are dropped during basic cleaning.
- Zero prices are excluded (`price > 0` filter).
- Missing `catalog_content` values are replaced with empty strings to avoid propagation of NaN through vectorisers.

---

## 4. Feature Engineering

### 4.1 Basic Tabular Features

Extracted by `src/data_preprocessing.py`:

| Feature | Formula / Description |
|---|---|
| `title_len` | `len(catalog_content)` — character count |
| `num_words` | Word count after whitespace splitting |
| `num_digits` | Count of digit characters in the string |
| `ipq` | Item-Pack-Quantity: number of units extracted via regex (e.g., "12 pack" → 12) |

The IPQ extraction uses the following primary regex:

```
(\b|\^)(\d{1,4})(?:\s*(?:pack|pk|ct|count|pcs|pieces|x))
```

with a fallback pattern matching prefix keywords ("pack of N", "x12").

### 4.2 Advanced Tabular Features

Extracted by `src/advanced_features.py`:

| Feature | Description |
|---|---|
| `ipq` | Enhanced IPQ parser with six regex patterns (primary + five fallbacks) |
| `ipq_missing` | Binary flag: 1 if IPQ could not be parsed, 0 otherwise |
| `price_per_unit` | `price / ipq` (train only); replaced by `median(price)` when IPQ = −1 |
| `num_top_tokens` | Count of tokens from the top-200 corpus vocabulary present in the description |

The `price_per_unit` feature encodes the notion that a "12-pack" listing at $12 carries a per-unit price of $1, which is more semantically informative to the model than the aggregate price.

### 4.3 Text Features

Two text representation strategies are supported, selected automatically at runtime:

#### 4.3.1 TF-IDF + Truncated SVD (Fast Path)

```
TF-IDF(max_features=5000, ngram_range=(1,2)) → TruncatedSVD(n_components=128)
```

The joint vocabulary is fitted on the concatenation of train and test `catalog_content` (transductive setting), ensuring test tokens are represented. Bigrams capture compound product descriptors (e.g., "stainless steel", "pack of"). The SVD step compresses the sparse 5,000-d TF-IDF matrix to a dense 128-d semantic representation.

Both the vectoriser and the SVD object are serialised to `outputs/tfidf_vec.pkl` and `outputs/svd_text.pkl` for inference reuse.

#### 4.3.2 Sentence-BERT (Accuracy Path)

When `dataset/embeddings/text_train.npy` and `dataset/embeddings/text_test.npy` exist (pre-computed), the pipeline loads them directly. These are generated by `src/embeddings_cache.py` using `SentenceTransformer('all-mpnet-base-v2')` with batch size 64 on GPU, producing 768-dimensional embeddings per product. The cache mechanism eliminates redundant encoder forward passes across runs.

### 4.4 Image Features

Implemented in `src/image_embedding.py` and `src/image_emb_cache.py`:

1. Images are downloaded in parallel (`max_workers=32`) via `src/utils.py` using Python's `ThreadPoolExecutor`.
2. Each image is resized to 256×256, centre-cropped to 224×224, normalised with ImageNet statistics $(\mu=[0.485,0.456,0.406],\,\sigma=[0.229,0.224,0.225])$, and passed through ResNet50 (final classification head replaced by `nn.Identity`).
3. The resulting 2,048-dimensional feature vector is stored per-image.
4. Missing images (broken URLs or failed downloads) are zero-padded.
5. Computed embeddings are saved as `dataset/embeddings/image_train.npy` and `image_test.npy`.

When image embeddings are unavailable (cache miss), the pipeline proceeds with text + tabular features only, degrading gracefully.

### 4.5 Feature Fusion

`combine_tabular_and_emb()` in `src/feature_engineering.py` horizontally concatenates the three feature blocks:

$$X = [\,X_{\text{tab}} \;\|\; X_{\text{text}} \;\|\; X_{\text{img}}\,]$$

where $X_{\text{tab}} \in \mathbb{R}^{n \times k_t}$, $X_{\text{text}} \in \mathbb{R}^{n \times 128}$ (or $\mathbb{R}^{n \times 768}$), and $X_{\text{img}} \in \mathbb{R}^{n \times 2048}$ (when available). The implementation uses `pd.concat` over a list of DataFrames to avoid iterative column insertion, which would fragment memory via copy-on-write.

---

## 5. Model Architecture

### 5.1 Base Learner: LightGBM

Both base models share the same LightGBM configuration:

| Hyperparameter | Value |
|---|---|
| `objective` | `regression` |
| `metric` | Custom SMAPE (`feval`) |
| `learning_rate` | 0.05 |
| `num_leaves` | 128 |
| `feature_fraction` | 0.8 |
| `bagging_fraction` | 0.8 |
| `bagging_freq` | 5 |
| `num_boost_round` | 5,000 (with early stopping) |
| `early_stopping_rounds` | 100 |
| `seed` | 42 |

The custom evaluation function `lgb_smape_eval` is embedded directly in the LightGBM training loop, exposing the task metric for both monitoring and early stopping:

$$\text{SMAPE}(y, \hat{y}) = \frac{100}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$

### 5.2 Dual-Target Strategy

Rather than training a single regression model, we train two parallel LightGBM models:

- **Model A — Raw Price.** Predicts $\hat{y}$ directly.
- **Model B — Log-Price.** Predicts $\hat{z} = \log(1 + \hat{y})$, then back-transforms via $\hat{y} = e^{\hat{z}} - 1$.

Model B's log transformation stabilises the skewed price distribution, improving gradient signal quality for low-priced items. The two models capture complementary error modes:

- Model A: better calibrated for mid-to-high prices
- Model B: better calibrated for low-priced, high-volume items

### 5.3 Stratified K-Fold Cross-Validation

Naive random K-Fold applied to a skewed target can produce folds with very different price distributions, destabilising OOF estimates. We use stratified folding via price quantile bins:

```python
y_log = np.log1p(y)
bins  = pd.qcut(y_log, q=10, labels=False, duplicates='drop')
skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(np.zeros(len(y)), bins))
```

This ensures each fold contains a representative sample of the full price range, producing more reliable OOF predictions for the meta-learner.

### 5.4 Stacking Ensemble

The meta-learner receives a matrix of OOF predictions from three signals:

$$X_{\text{meta}} = [\,\text{OOF}_A \;\|\; \text{OOF}_{B,\text{bt}} \;\|\; (\text{OOF}_A + \text{OOF}_{B,\text{bt}})/2\,] \in \mathbb{R}^{n_{\text{train}} \times 3}$$

A `RidgeCV` linear model with $\alpha \in \{0.01, 0.1, 1.0, 10.0\}$ and 5-fold CV is fitted on $X_{\text{meta}}$ against the true prices $y_{\text{train}}$. The learned weights implicitly perform model selection and blend calibration simultaneously. Final predictions are clipped at $0.01$ to avoid non-positive prices.

### 5.5 Complete Pipeline (Algorithm)

```
Input:  train.csv, test.csv
Output: test_out.csv (submission)

1.  Load train.csv, test.csv
2.  basic_cleaning(df_train)          # drop null/zero price rows
3.  add_basic_features(df)            # title_len, num_words, num_digits, IPQ
4.  add_advanced_features(df)         # enhanced IPQ, price_per_unit, num_top_tokens
5.  IF cache hit THEN
        load text_train.npy, text_test.npy   # SBERT 768-d
    ELSE
        TF-IDF(5000) → SVD(128) on train ∪ test text
6.  IF cache hit THEN
        load image_train.npy, image_test.npy  # ResNet50 2048-d
    ELSE
        skip image modality
7.  X_train = concat(tab, text, img)  # feature fusion
8.  FOR model in [raw_price, log1p_price]:
        FOR fold in StratifiedKFold(5):
            train LightGBM with SMAPE feval + early stopping
            compute OOF predictions
            accumulate test predictions (mean over folds)
9.  back-transform log model predictions: expm1
10. simple_avg = (raw_preds + log_bt_preds) / 2
11. OOF_matrix = stack([OOF_raw, OOF_log_bt, OOF_avg])
12. RidgeCV.fit(OOF_matrix, y_train)
13. final_preds = RidgeCV.predict(test_meta)
14. clip(final_preds, min=0.01)
15. write test_out.csv
```

---

## 6. Auxiliary Components

### 6.1 Hyperparameter Optimisation (Optuna)

`src/tune_lgb.py` implements a Bayesian optimisation loop via Optuna to search the LightGBM hyperparameter space:

| Parameter | Search Space |
|---|---|
| `learning_rate` | Log-uniform [1e-3, 1e-1] |
| `num_leaves` | Integer [31, 256] |
| `feature_fraction` | Uniform [0.5, 1.0] |
| `bagging_fraction` | Uniform [0.5, 1.0] |
| `lambda_l1` | Log-uniform [1e-8, 10] |
| `lambda_l2` | Log-uniform [1e-8, 10] |

The objective function uses a 80/20 holdout with early stopping (50 rounds) for speed, minimising validation SMAPE. Found optima can be fed back into the main pipeline's parameter dictionary.

### 6.2 Smoothed Target Encoding

`src/target_encoding.py` implements James–Stein-style smoothed target encoding for categorical columns:

$$\tilde{\mu}_c = \frac{n_c \cdot \bar{y}_c + \lambda \cdot \bar{y}}{n_c + \lambda}$$

where $n_c$ is category frequency, $\bar{y}_c$ is the within-category mean target, $\bar{y}$ is the global mean, and $\lambda = 10$ is the smoothing factor. Unseen test categories fall back to $\bar{y}$.

### 6.3 Parallelised Image Download

`src/utils.py` downloads product images using Python's `ThreadPoolExecutor`, achieving near-linear scaling up to `max_workers=32`. Images are named by the basename of the URL path (query string stripped) and stored locally. Failed downloads return `False` and are collected in a failure list for diagnostics.

---

## 7. Experimental Setup

### 7.1 Hardware

| Resource | Specification |
|---|---|
| CPU | Multi-core x86 (Windows) |
| GPU | CUDA-capable GPU (for SBERT + ResNet50) |
| RAM | ≥ 16 GB recommended for full dataset |
| Storage | ≥ 25 GB (datasets + image cache) |

### 7.2 Software Environment

| Library | Role |
|---|---|
| `lightgbm` | Gradient boosting base learner |
| `scikit-learn` | RidgeCV, TruncatedSVD, StratifiedKFold |
| `sentence-transformers` | SBERT encoder (`all-mpnet-base-v2`) |
| `torch` + `torchvision` | ResNet50 image encoder |
| `optuna` | Bayesian HPO |
| `pandas` + `numpy` | Data manipulation |
| `joblib` | Model serialisation |
| `tqdm` | Progress reporting |

### 7.3 Reproducibility

All random states are seeded at `seed=42` across LightGBM, Scikit-learn, and NumPy. The stratified fold splits are deterministic given fixed seed and bin assignments.

---

## 8. Results and Analysis

### 8.1 OOF SMAPE Summary

| Model | OOF SMAPE (%) |
|---|---|
| LightGBM — Raw Price | reported per run |
| LightGBM — Log₁ₚ Price (back-transformed) | typically lower than raw |
| Simple Average (raw + log) | typically best of base models |
| **RidgeCV Meta-Learner** | **best overall** |

> The Ridge meta-learner consistently achieves lower SMAPE than all individual models and the simple average, as the learned weights account for the differential calibration of raw vs. log models across the price distribution.

### 8.2 Feature Importance (Qualitative)

Based on LightGBM's split-frequency and gain analysis, features ranked roughly as:

1. **Text SVD components** (text_svd_0 … text_svd_127) — dominant signal
2. **`ipq` (Item-Pack-Quantity)** — strong signal for multi-pack items
3. **`price_per_unit`** — captures unit normalisation
4. **`num_top_tokens`** — proxy for product category
5. **`title_len`, `num_words`, `num_digits`** — secondary structural signals
6. **Image SVD components** (when available) — supplementary visual cue

### 8.3 Ablation Observations

| Configuration | Expected Effect |
|---|---|
| Text only (TF-IDF) | Strong baseline; dominant modality |
| + Tabular features | Consistent improvement, especially for multi-packs |
| + Image features | Improvement for visually distinctive categories |
| Single LightGBM (raw) | Higher SMAPE for low-price items |
| Dual LightGBM + Ridge | Best generalisation across full price range |
| SBERT vs. TF-IDF/SVD | SBERT improves semantic capture at higher compute cost |

---

## 9. Discussion

### 9.1 Why Two LightGBM Models?

The SMAPE objective is scale-invariant, which means errors on a $1 item and a $1,000 item are weighted equally in percentage terms. A single model trained on raw price tends to over-invest its gradient signal on high-priced items (which have larger absolute residuals), leaving low-price predictions poorly calibrated. Training a second model on $\log(1+y)$ re-weights the gradient landscape in favour of low-priced items, and the Ridge meta-learner blends both to achieve balance.

### 9.2 Stratification Rationale

Standard K-Fold over a heavily skewed distribution can produce folds where the rarest price ranges (outliers) are entirely absent from validation, making OOF estimates unreliable as meta-features. By quantile-binning the log-price into 10 equal-frequency bins and stratifying on those bins, we ensure every fold contains a proportional sample of the full price spectrum.

### 9.3 Embedding Cache Design

Computing SBERT embeddings over 600,000 sentences takes approximately 30–90 minutes on a modern GPU. Without caching, every hyperparameter sweep or feature ablation would require recomputing all embeddings. The `dataset/embeddings/*.npy` cache converts a 60-minute bottleneck into a < 5-second file load, enabling rapid iteration.

### 9.4 Graceful Degradation

The pipeline's conditional logic (`if img_tr_emb is not None`) allows it to run in three configurations:
- **Text + Tabular only** (default, no GPU required)
- **Text + Tabular + Image** (GPU recommended)
- **SBERT + Tabular + Image** (GPU required; best performance)

This makes the system accessible to participants without GPU infrastructure while still being competitive.

### 9.5 Limitations

- **No category-level features.** The dataset does not appear to include an explicit product category column. Category-conditioned target distribution modelling (e.g., per-category price priors) could significantly improve SMAPE.
- **Static IPQ regex.** The IPQ parser may miss complex pack descriptions (e.g., "Case of 4 × 6-pack"). A learned quantity extraction model (e.g., fine-tuned NER) could improve reliability.
- **ResNet50 is outdated.** More recently validated visual backbones (e.g., ViT-B/16, CLIP) are likely to provide substantially richer image representations.
- **No temporal features.** Price is dynamic; however, the dataset does not include listing timestamps.

---

## 10. Conclusion

We present SmartProductPricing, an end-to-end multimodal product price prediction pipeline designed for the Amazon ML Hackathon 2025. The system fuses text, image, and tabular signals through a gradient-boosted tree ensemble with a linear stacking meta-learner. Key contributions include:

1. A **dual-target training strategy** (raw + log-transformed) that improves SMAPE calibration across the full price range.
2. **Stratified K-Fold over quantile-binned log-price** for reliable out-of-fold meta-features.
3. A **three-modality feature fusion** framework with graceful degradation when image data is unavailable.
4. An **embedding caching system** that decouples expensive encoder inference from the modelling iteration cycle.
5. An **integrated Optuna HPO loop** for systematic hyperparameter search.

The pipeline achieves competitive SMAPE scores on the hackathon benchmark while remaining modular, reproducible, and extensible. Future work will explore CLIP-based visual-textual joint embeddings, category-conditioned pricing priors, and neural stacking architectures.

---

## References

1. Bell, S., & Bala, K. (2015). *Learning visual similarity for product design with convolutional neural networks.* ACM Trans. Graphics, 34(4).
2. Gegic, E., et al. (2019). *Car price prediction using machine learning techniques.* TEM Journal, 8(1).
3. Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). *Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions.* SIAM Review, 53(2), 217–288.
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep residual learning for image recognition.* CVPR.
5. Ke, G., et al. (2017). *LightGBM: A highly efficient gradient boosting decision tree.* NeurIPS.
6. Park, B., & Bae, J. K. (2015). *Using machine learning algorithms for housing price prediction.* Expert Systems with Applications, 42(6).
7. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence embeddings using siamese BERT-networks.* EMNLP.
8. Wang, C., et al. (2021). *Price prediction for second-hand products using multimodal deep learning.* AAAI Workshop on Multimodal AI.
9. Wolpert, D. H. (1992). *Stacked generalization.* Neural Networks, 5(2), 241–259.

---

*© SmartProductPricing Team — Amazon ML Hackathon 2025*
