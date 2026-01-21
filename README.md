# Blue-Light Blocking Glasses Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **CSE 424: Pattern Recognition Project**  
> Department of Computer Science and Engineering, BRAC University

## 📋 Table of Contents
- [Overview](#overview)
- [Research Objectives](#research-objectives)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Implementation](#implementation)
- [Results](#results)
- [Key Findings](#key-findings)
- [Installation](#installation)
- [Usage](#usage)
- [Team Members](#team-members)
- [Acknowledgements & Supervision](#acknowledgements--supervision)
- [References](#references)

## 🔬 Overview

This project investigates the spectrophotometric properties of commercially available blue-light blocking lenses using machine learning regression models. Blue light, emitted by electronic devices, LED lights, and sunlight, can disrupt circadian rhythms and affect sleep quality by suppressing melatonin production through activation of intrinsically photosensitive retinal ganglion cells (ipRGCs).

Our research analyzes **50 commercial blue-blocking lenses** under **5 different lighting conditions** to:
- Quantify their effectiveness in filtering blue light (380-500nm range)
- Evaluate transmission specificity between circadian-proficient (455-560nm) and non-proficient wavelengths
- Develop predictive models for spectrophotometric characteristics
- Provide evidence-based recommendations for optimal lens selection

## 🎯 Research Objectives

1. **Spectrophotometric Analysis**: Measure transmission spectra across the visible spectrum (380-780nm) for various blue-blocking lenses under diverse lighting conditions
2. **Machine Learning Prediction**: Develop and compare regression models (KNN, SVM, Linear Regression) to predict lens performance
3. **Circadian Impact Assessment**: Evaluate transmission specificity to determine effectiveness in blocking circadian-proficient light while maintaining visibility
4. **Practical Recommendations**: Identify optimal lens types for regulating sleep-wake cycles based on empirical evidence

## 🔍 Methodology

### Data Collection
- **Sample Size**: 50 commercial blue-blocking lenses
- **Light Sources**: 
  - Natural sunlight
  - Fluorescent overhead luminaire
  - Incandescent lamp
  - Blue LED array
  - Computer tablet display
- **Measurement Range**: 380-780nm (visible spectrum)
- **Metrics**: Absolute irradiance, percentage transmission, transmission specificity

### Lens Categorization
Lenses were grouped by tint color:
- Clear/Minimal tint
- Yellow/Amber tint
- Orange tint
- Red tint
- Reflective blue coating

### Feature Engineering
Key features analyzed:
- **Lens Material**: Polycarbonate, CR-39, glass
- **Coating Type**: Anti-reflective, blue-blocking, UV-filtering
- **Tint Properties**: Color, density, transmission spectrum
- **Manufacturing Process**: Coating vs. substrate dyeing

## 📊 Dataset

The dataset comprises spectrophotometric measurements for 50 commercial lenses:

| Feature | Description | Type |
|---------|-------------|------|
| Lens ID | Unique identifier | Categorical |
| Tint Type | Color classification | Categorical |
| Circadian Transmission | Light transmission in 455-560nm range | Continuous |
| Non-Circadian Transmission | Light transmission in 380-454nm & 561-780nm | Continuous |
| Transmission Specificity | Ratio metric for selective filtering | Continuous |
| Light Source | Testing condition | Categorical |

**Data Preprocessing**:
- Missing value imputation
- Feature normalization/standardization
- Categorical encoding
- Train-test split (70-30 ratio)

## 💻 Implementation

### Technologies Used
- **Programming Language**: Python 3.8+
- **Machine Learning**: Scikit-learn
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Data Cleaning**: OpenRefine

### Models Implemented

#### 1. K-Nearest Neighbors (KNN)
```python
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=optimal_k)
knn_model.fit(X_train, y_train)
```
- **Optimal K Selection**: Cross-validation and grid search
- **Distance Metric**: Euclidean distance
- **Best Performance**: Achieved through hyperparameter tuning

#### 2. Support Vector Machine (SVM)
```python
from sklearn.svm import SVR
svm_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svm_model.fit(X_train, y_train)
```
- **Kernel**: Radial Basis Function (RBF)
- **Regularization**: C parameter optimization

#### 3. Linear Regression
```python
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
```
- **Baseline Model**: Provides linear relationship interpretation

### Evaluation Metrics

**Mean Squared Error (MSE)**:
```
MSE = (1/n) * Σ(yi - ŷi)²
```

**Root Mean Squared Error (RMSE)**:
```
RMSE = √MSE
```

**Mean Absolute Error (MAE)**:
```
MAE = (1/n) * Σ|yi - ŷi|
```

**R-squared (R²)**:
```
R² = 1 - [Σ(yi - ŷi)² / Σ(yi - ȳ)²]
```

## 📈 Results

### Model Performance Comparison

| Model | MSE | RMSE | MAE | R² Score |
|-------|-----|------|-----|----------|
| **K-Nearest Neighbors (KNN)** | **35.059** | **5.921** | **3.844** | **0.914** ✓ |
| Linear Regression | 82.907 | 9.105 | 1.399 | 0.797 |
| Support Vector Machine (SVM) | 211.039 | 14.527 | 10.365 | 0.483 |

### Performance Insights

**K-Nearest Neighbors (Winner)**:
- ✅ **Highest R² Score (91.4%)**: Explains 91.4% of variance in spectrophotometric data
- ✅ **Lowest Error Metrics**: Best prediction accuracy across all measurements
- ✅ **Robust Performance**: Consistent results across different lighting conditions
- **Recommendation**: Optimal model for predicting blue-light blocking effectiveness

**Linear Regression (Moderate)**:
- ⚠️ Reasonable R² (79.7%) with balanced error metrics
- ⚠️ Lower MAE indicates good average performance
- ⚠️ Suitable for understanding linear relationships between features

**Support Vector Machine (Underperformer)**:
- ❌ Lowest R² (48.3%) indicates poor fit
- ❌ Highest error metrics suggest significant prediction errors
- ❌ May require extensive hyperparameter tuning or different kernel

## 🔑 Key Findings

### Lens Performance by Tint

1. **Red-Tinted Lenses**:
   - ✓ Transmitted the **least circadian-proficient light** (455-560nm)
   - ✓ Most effective for blocking blue light activation of ipRGCs
   - ✗ Significant color distortion may impact daily usability

2. **Orange-Tinted Lenses**:
   - ✓ **Highest transmission specificity** in normal daylight
   - ✓ Optimal balance between blue light blocking and color perception
   - ✓ **Best recommendation** for regulating circadian sleep-wake rhythms
   - ✓ Minimal visual distortion while maintaining protection

3. **Yellow/Amber-Tinted Lenses**:
   - ≈ Similar circadian-proficient light transmission to orange lenses
   - ✗ Higher non-circadian light transmission
   - ✗ Lower transmission specificity compared to orange

4. **Reflective Blue Lenses**:
   - ✗ Highest transmission of circadian-proficient light
   - ✗ Least effective for circadian rhythm regulation
   - ✗ Not recommended for sleep improvement purposes

### Clinical Implications

- **Insomnia Treatment**: Blue-blocking glasses show promise as non-pharmacological intervention
- **Delayed Sleep Phase Disorder**: Orange-tinted lenses can advance circadian rhythms
- **Evening Light Exposure**: Wearing blue-blockers 2-3 hours before bedtime may improve melatonin production
- **Shift Workers**: Beneficial for managing irregular sleep schedules

## 🚀 Installation

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
```

### Clone Repository
```bash
git clone https://github.com/tareqaziz825/BRACU-CSE-424-PATTERN-RECOGNITION.git
cd BRACU-CSE-424-PATTERN-RECOGNITION
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 📖 Usage

### Running the Analysis
```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv('blue_blocker_data.csv')

# Prepare features and target
X = data.drop('spectrometric_value', axis=1)
y = data['spectrometric_value']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)

# Make predictions
predictions = knn_model.predict(X_test)

# Evaluate
rmse = mean_squared_error(y_test, predictions, squared=False)
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")
```

### Visualizing Results
```python
import matplotlib.pyplot as plt

# Scatter plot: Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.xlabel('Actual Spectrometric Values')
plt.ylabel('Predicted Spectrometric Values')
plt.title('KNN Model: Actual vs Predicted')
plt.grid(True, alpha=0.3)
plt.show()
```

## 👥 Team Members

| Name | Contributions |
|------|------|
| **Mohammod Tareq Aziz Justice** | Team Lead, Model Development |
| **Driciru Fiona** | Data Collection & Processing |
| **Ajani Denish** | Feature Engineering |
| **Sudirgha Chakma** | Visualization & Analysis

## 🙏 Acknowledgements & Supervision

### Instructor
**Annajiat Alim Rasel**  
Senior Lecturer  
Department of Computer Science and Engineering  
BRAC University  
📧 Email: annajiat@bracu.ac.bd

### Course Support
**Sadiul Arefin Rafi**  
Adjunct Lecturer  
Department of Computer Science and Engineering  
BRAC University  
📧 Email: ext.sadiul.rafi@bracu.ac.bd

We express our sincere gratitude to our instructors for their invaluable guidance, continuous support, and expertise throughout this research project. Their insights into pattern recognition methodologies and machine learning best practices were instrumental in shaping our work.

Special thanks to BRAC University for providing the resources and academic environment necessary for conducting this research.

## 📚 References

1. Mason, B. J., Tubbs, A. S., Fernandez, F.-X., & Grandner, M. A. (2022). "Spectrophotometric properties of commercially available blue blockers across multiple lighting conditions." figshare.

2. Dange, D., et al. (2021). "Evening wear of blue-blocking glasses for sleep and mood disorders: a systematic review." https://doi.org/10.1080/07420528.2021.1930029

3. Skrede, S., et al. (2016). "Blue-blocking glasses as additive treatment for mania: a randomized placebo-controlled trial." https://doi.org/10.1111/bdi.12390

4. Sasseville, A., Paquet, N., Sévigny, J., & Hébert, M. (2006). "Blue blocker glasses impede the capacity of bright light to suppress melatonin production." https://doi.org/10.1111/j.1600-079X.2006.00332.x

5. Alzahrani, H. S., Khuu, S. K., & Roy, M. (2021). "Modelling the effect of commercially available blue-blocking lenses on visual and non-visual functions." https://doi.org/10.1111/cxo.12959

6. Lawrenson, J. G., Hull, C. C., & Downie, L. E. (2017). "The effect of blue-light blocking spectacle lenses on visual performance, macular health and the sleep-wake cycle: a systematic review." https://doi.org/10.1111/opo.12406

7. Bigalke, J. A., et al. (2021). "Effect of evening blue light blocking glasses on subjective and objective sleep in healthy adults: A randomized control trial." Sleep Medicine, 7(4), 485-490.

8. Esaki, Y., et al. (2016). "Wearing blue light-blocking glasses in the evening advances circadian rhythms in patients with delayed sleep phase disorder." https://doi.org/10.1080/07420528.2016.1194289

9. Gregoire, A. C., et al. (2023). "Blue Light Exposure: Ocular Hazards and Prevention—A Narrative Review." https://doi.org/10.1007/s40123-023-00675-3

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or collaboration opportunities, please reach out to:
- **Project Repository**: [GitHub](https://github.com/tareqaziz825/BRACU-CSE-424-PATTERN-RECOGNITION)
- **Paper**: [IEEE Format PDF](https://github.com/tareqaziz825/BRACU-CSE-424-PATTERN-RECOGNITION/blob/main/CSE%20424%20Project%20Paper%20IEEE.pdf)

---

**Note**: This research was conducted as part of the CSE 424: Pattern Recognition course at BRAC University. The findings provide valuable insights into blue-light filtering effectiveness but should not replace professional medical advice for sleep or vision-related conditions.

**Keywords**: Blue Light Blocking, Machine Learning, Spectrophotometry, K-Nearest Neighbors, Circadian Rhythm, Sleep Health, Pattern Recognition, Regression Analysis
