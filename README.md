# NDIS

**The Normal Distributions Indistinguishability Spectrum and its Application to Privacy-Preserving Machine Learning**

This repository provides a proof-of-concept implementation of the generic way of analytically computing the DP-properties for algorithms whose outputs are Gaussian. The method is proposed in our paper:

**[The Normal Distributions Indistinguishability Spectrum and its Application to Privacy-Preserving Machine Learning](https://arxiv.org/pdf/2309.01243)**

In the repository, we implement the NDIS estimator, which takes as input the parameters of a pair of Gaussian and neumerically estimate the DP-distance between this pair of Gaussian. We also implement the DP Least Square and DP Random projection mechanisms which derived from the NDIS analysis, as shown in our paper. Moreover, we use the black-box eureka DP estimator (proposed in paper) to show the asymptotic privacy analysis of the Approximate Least Square  (an efficient alternative of the DP Least Square) tightely matches its empirical privacy guarantee. 
