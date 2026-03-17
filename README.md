## Wildfire Smoke Classification

A comparative analysis of machine learning models (LDA, Random Forest, TabNet) for classifying wildfire fuel types based on their chemical signatures. Following the methodology of Stamatis & Barsanti (2022), this project demonstrates that a 1936 statistical method can match state-of-the-art deep learning on small, linearly separable datasets—challenging the assumption that newer is always better.

## Biological Problem and Motivation

Wildfire smoke composition varies by fuel type, with different tree species emitting distinct chemical fingerprints called **monoterpenoids**. Accurately identifying fuel sources from smoke samples is critical for fire management to predict fire behavior based on available fuels and air quality modeling to understand health impacts of smoke exposure

**The Challenge:** Can we classify fuel types (Pine vs Fir) from their monoterpenoid profiles using only chemical data? More importantly, do we need cutting-edge deep learning, or will classical statistical methods suffice?

## Data Source and Usage

**Dataset:** `Sample_data.csv` (included in this repository)

**Source:** GC×GC-TOFMS (Two-dimensional Gas Chromatography - Time of Flight Mass Spectrometry) analysis of wildfire smoke samples from controlled laboratory burns.

**Original Study:** Stamatis, C. & Barsanti, K. (2022). Development and application of a supervised pattern recognition algorithm for identification of fuel-specific emissions profiles. *Atmospheric Measurement Techniques*, 15, 2591–2606. https://doi.org/10.5194/amt-15-2591-2022

**Data Characteristics:**
- **Initial samples:** 50 smoke samples from controlled burns
- **Fuel species:** 6 species (ponderosa pine, lodgepole pine, Douglas fir, subalpine fir, Engelmann spruce, manzanita)
- **Fuel families:** Grouped into 4 families (Pine, Fir, Spruce, Shrub)
- **Chemical measurements:** ~97 monoterpenoid compound concentrations per sample
- **Final dataset:** 20 samples (10 Pine, 10 Fir) after quality filtering

**Data Processing:**
- Encoding fix required (Greek letters: α-pinene, β-pinene → use `encoding='latin-1'`)
- Detection rate filter (≥30% of compounds detected)
- Families with <3 samples removed (Spruce and Shrub dropped)

## Computational Approach and Workflow

This analysis follows a classic chemometrics pipeline combining statistical feature selection, dimensionality reduction, and supervised classification:

### Workflow Steps

**1. Data Cleaning**
- Handle UTF-8 encoding errors (Greek letters in compound names)
- Calculate detection rate per sample (% of compounds detected and >0)
- Filter samples with <30% detection
- Remove fuel families with <3 samples for statistical validity

**2. Feature Selection via ANOVA**
- Standardize compound concentrations (z-score normalization)
- Calculate F-ratio for each compound (between-group variance / within-group variance)
- Select top 5 compounds by F-statistic
- **Selected compounds:** Camphene (F=134.0), Tricyclene (F=115.3), Myrtenal (F=79.0), β-Pinene (F=77.1), 3-Carene (F=61.3)

**3. Dimensionality Reduction via PCA**
- Reduce 5 compounds → 2 principal components
- PC1 explains ~87% variance (primary discriminant)
- PC2 explains ~6% variance (secondary pattern)
- Total: 93% variance captured in 2D space

**4. Classification Model Comparison**
- **LDA (Linear Discriminant Analysis, 1936):** Finds optimal linear boundary, no hyperparameters
- **Random Forest (2001):** Ensemble of 100 decision trees, handles non-linearity
- **TabNet (2020):** Deep learning with attention mechanisms, designed for tabular data
- All models trained on **identical 70/30 split** (random_state=42, stratified)

**Random state = 42:** A seed number ensuring reproducible train/test splits. Without it, you'd get different random splits each run, making results non-comparable.

## 🧗 Route Design and Completion Status

This analysis is structured as a "climbing route" with progressive exercises:

| Exercise | Task | Status |
|----------|------|--------|
| **Exercise 0** | Load data, handle encoding errors | Complete |
| **Exercise 1** | Create fuel family groupings | Complete |
| **Exercise 2** | Filter by detection rate (≥30%) | Complete |
| **Exercise 3** | ANOVA feature selection (top 5 compounds) | Complete |
| **Exercise 4** | PCA visualization (2D reduction) | Complete |
| **Exercise 5** | Train LDA, Random Forest, TabNet | Complete |
| **Exercise 6** | Design experiment where TabNet wins | Optional |
| **Exercise 7** | Reflection and interpretation | Complete |

**Educational Goal:** Demonstrate that the simplest model solving your problem is the right model—not the newest or most complex.

## 📈 Results Summary

### Model Performance

| Model | Year | Training Accuracy | Test Accuracy | Overfitting Gap |
|-------|------|-------------------|---------------|-----------------|
| **LDA** | 1936 | 100.0% | 100.0% | 0.0% |
| **Random Forest** | 2001 | 100.0% | 100.0% | 0.0% |
| **TabNet** | 2020 | 100.0% | 100.0% | 0.0% |

**Key Finding:** All three models achieved perfect classification. The 84-year-old linear method matched state-of-the-art deep learning.

### Key Figures

**Figure 1: Feature Selection (ANOVA F-ratios)**
![Top 20 compounds ranked by F-ratio, with top 5 highlighted]

**Figure 2: PCA Visualization**
![2D scatter plot showing clear linear separation between Pine (blue) and Fir (orange) along PC1 axis]
- PC1 explains 87.3% variance
- PC2 explains 5.5% variance
- Linearly separable clusters

**Figure 3: TabNet Feature Importance**
![Attention weights showing PC1 dominance]
- PC1: 94.6% (TabNet agrees PC1 contains primary signal)
- PC2: 5.4%

**Figure 4: Decision Boundaries**
![Side-by-side comparison showing all three models draw nearly identical linear boundaries]

### Training Behavior

**TabNet early stopping:** Training stopped at **epoch 0-1** with validation accuracy already at 100%. This indicates the classification problem is trivially easy—even random initialization achieved perfect performance.

## Interpretation, Limitations, and Next Steps

### Why Did LDA Tie TabNet?

**Three critical factors:**

1. **Small sample size:** Only 14 training samples—orders of magnitude below what deep learning needs (typically 1,000+)
2. **Linear separability:** Pine and Fir are separated by a simple straight line in PC space
3. **Problem too easy:** TabNet stopped immediately because there was nothing to learn beyond random initialization

**Analogy:** Using TabNet here is like using a supercomputer to add 2+2. The tool is powerful, but the problem doesn't need that power.

### When Would TabNet Outperform LDA?

TabNet would win when:
- **Sample size:** >1,000 samples (ideally 10,000+)
- **Non-linear patterns:** Complex interactions between features
- **High dimensionality:** 50-200+ features with intricate relationships
- **Missing data:** TabNet handles this natively via attention mechanisms

### Limitations

1. **Small dataset (n=20)** limits generalizability and prevented analysis of all fuel families
2. **Only 2 fuel families** analyzed (Spruce and Shrub dropped due to <3 samples after filtering)
3. **Laboratory burns** may not perfectly reflect real wildfire conditions (moisture, wind, mixed fuels)
4. **TIC normalization** used instead of CO normalization (paper's method)
5. **Single study replication** - needs validation on independent datasets

### Next Steps

1. **Expand dataset:** Collect more samples to include Spruce and Shrub families (target: 50+ samples per family)
2. **Field validation:** Test on BFRS field campaign data from real wildfires
3. **Environmental covariates:** Add temperature, humidity, burn rate as predictors
4. **Multi-region validation:** Test model on fires from different geographic areas
5. **Ensemble approach:** Combine predictions from family-specific models
6. **Time-series analysis:** Track how emissions change throughout burn progression

## Reproducibility Instructions

### Prerequisites

**Required packages:**
```bash
pip install pandas numpy scipy scikit-learn matplotlib pytorch-tabnet jupyter
```

### Step-by-Step Instructions

**1. Clone this repository:**
```bash
git clone https://github.com/BryanRinde/Smoke-Analysis.git
cd Smoke-Analysis
```
**2. Verify data file is present:**
```bash
ls Sample_data.csv
```
*(File is included in repository—no separate download needed)*

**3. Open the analysis notebook:**
- Navigate to `Smoke_Analysis_All.ipynb` in the Jupyter interface
- Click to open

**4. Run the analysis:**
- **Option A:** Run all cells at once: `Cell` → `Run All`
- **Option B:** Run cells sequentially to follow along: `Shift + Enter` for each cell

### Expected Outputs

The notebook generates these artifacts:

**Figures:**
- `f_ratio_analysis.png` - Feature selection results (top 20 compounds by F-ratio)
- `pca_variance_explained.png` - Scree plot showing variance by principal component
- `pca_pc1_vs_pc2.png` - 2D fuel family separation (main visualization)
- `pca_pc1_vs_pc4.png` - Alternative PC combination (paper method)
- `pca_biplot.png` - Samples and variable loadings combined
- `kmeans_elbow_plot.png` - Optimal cluster number determination
- `kmeans_clustering.png` - Unsupervised clustering results
- `lda_decision_boundary.png` - LDA classification with decision boundary
- `lda_probability_distributions.png` - Prediction probability distributions
- `lda_vs_random_forest.png` - LDA and Random Forest decision boundaries comparison
- `probability_comparison.png` - LDA vs Random Forest probability heatmaps
- `tabnet_feature_importance.png` - TabNet attention weights (PC1 vs PC2)
- `tabnet_training_curves.png` - TabNet training loss and validation accuracy

**Data files:**
- `lda_classification_results.csv` - Per-sample predictions and probabilities
