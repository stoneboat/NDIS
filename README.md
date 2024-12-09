
# NDIS


**The Normal Distributions Indistinguishability Spectrum (NDIS) and its Application to Privacy-Preserving Machine Learning**

This repository provides a proof-of-concept implementation of the NDIS theory—an analytical computation of the differential privacy (DP) properties for algorithms whose output is arbitrary Gaussian. The repository also implement a DP Least Squares and a DP Random Projection mechanism, which are all derived from concrete NDIS-based analysis. The implementation for both general results and concrete DP algorithms follow our paper:

**[The Normal Distributions Indistinguishability Spectrum and its Application to Privacy-Preserving Machine Learning](https://arxiv.org/pdf/2309.01243)**


## Overview

Differential privacy (DP) ensures that an algorithm’s output does not reveal too much about any individual record in its input dataset. While the DP guarantees of the standard Gaussian mechanism are well-established, extending these results to mechanisms that produce Gaussian outputs with more complex, non-standard parameters remains challenging and has not been thoroughly explored. To address this gap, the Normal Distributions Indistinguishability Spectrum (NDIS) provides a systematic approach for directly computing privacy parameters (e.g., ε, δ) from the Gaussian parameters of such mechanisms, offering a clearer and more analytically grounded path for understanding their privacy properties.



**Key Features:**

- **NDIS Estimator:** Compute DP-distances between pairs of Gaussian distributions.
- **DP Least Squares (LS) Mechanisms:** Differentially private Least Squares mechanisms built from a NDIS-based analysis.
- **DP Random Projection (RP) Mechanisms:** Differentially private dimensionality reduction strategies built from a NDIS-based analysis.
- **Approximate Least Squares (ALS) Mechanism:** An efficient variant of the DP LS mechanism. In this repository, we instantiate the [Eureka](https://github.com/stoneboat/dp-estimation.git), a black-box DP estimator to  demonstrate that its asymptotic privacy analysis closely matches the empirical privacy guarantees.

## Repository Structure

```bash
NDIS/
├─ notebook/
│  ├─ ... (demonstrations of using the functionality in this repository)
│  ├─ ... (some comparison between the proposed LS and RP mechanisms and the literatures)
│
└─ src/
   ├─ Dataset/
   │  ├─ ... (data loading and preprocessing)
   │
   ├─ LS_mechanisms/
   │  ├─ ... (DP LS mechanism implementations)
   │
   ├─ RP_mechanisms/
   │  ├─ ... (DP RP mechanism implementations)
   │
   ├─ analysis/
   │  ├─ ALS_privacy_analysis.py  # Privacy analysis for Approximate LS
   │  ├─ NDIS_estimator.py        # Core NDIS estimation functions
   │  ├─ RP_privacy_analysis.py   # Privacy analysis for RP mechanisms
   │  ├─ accuracy_analysis.py     # Accuracy analyses
   │  ├─ commons.py               # Common helper functions
   │
   ├─ classifier/
   │  ├─ ... (Classifiers for instantiating the Eureka DP estimators)
   │
   ├─ estimator/
   │  ├─ ... (Eureka DP estimators for Random Projection and ALS algorithm)
   │
   ├─ utils/
   └─ ... (General utilities)
   ```

## Getting Started

### Prerequisites
- **Python Version:** Python 3.8+ is recommended.
- **Required Libraries:** Install the following common data science and machine learning libraries:
  - `numpy`, `scipy`, `scikit-learn`, `matplotlib`
  - `torch` (for Neural Network-based classifiers)

### Running the Examples
To explore the functionality and learn how to run the code, navigate to the `notebooks` folder and execute the provided Jupyter notebooks. 

## Citation

If you find this work helpful, please cite our paper: **[The Normal Distributions Indistinguishability Spectrum and its Application to Privacy-Preserving Machine Learning](https://arxiv.org/pdf/2309.01243)**.

## Contact 
For questions or comments, please open an issue on GitHub or contact the authors at ywei368@gatech.edu.
