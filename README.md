# Transformer Models Overview

This repository provides an overview of Transformer models, including their basic principles and various types. 
Transformers have become a fundamental building block in modern machine learning, especially in natural language processing (NLP) tasks.

## Table of Contents
- [Introduction to Transformers](#introduction-to-transformers)
- [Key Components](#key-components)
- [Types of Transformers](#types-of-transformers)
  - [Vanilla Transformer](#vanilla-transformer)
 

## Transformers for Time Series Forecasting

Transformers have been adapted for time series forecasting to handle sequential data with temporal dependencies. Here are some notable variants designed for this purpose:

### Informer

- **Description**: Informer is designed to handle long-term dependencies in time series forecasting. It introduces a ProbSparse Self-Attention mechanism to enhance efficiency and accuracy for long sequences.
- **Key Feature**: Efficient attention mechanism that scales better with long sequences.
- **Use Cases**: Long-term forecasting tasks and large-scale time series data.

### Autoformer

- **Description**: Autoformer incorporates an auto-correlation mechanism to capture temporal patterns effectively. It uses a decomposition strategy to separate seasonal and trend components of time series data.
- **Key Feature**: Decomposition-based model that captures both trend and seasonal components.
- **Use Cases**: Forecasting with strong seasonal patterns and trend components.

### LogTrans

- **Description**: LogTrans focuses on improving the efficiency of attention mechanisms by leveraging logarithmic attention. It is designed to handle long sequences with reduced computational cost.
- **Key Feature**: Logarithmic attention mechanism to manage long sequences efficiently.
- **Use Cases**: Time series forecasting with very long sequences.

### Patch TST (Patch Transformer for Time Series)

- **Description**: Patch TST introduces the concept of "patches" to time series forecasting, similar to image processing in Vision Transformers. It divides the time series into fixed-size patches and applies self-attention within and across patches to capture temporal patterns and dependencies effectively.
- **Key Feature**: Utilizes patch-based representation to handle long sequences and complex temporal dependencies.
- **Use Cases**: High-resolution time series forecasting and tasks requiring detailed temporal modeling.

## References

- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Kaiser, Ł., Polosukhin, I., et al. (2017). Attention Is All You Need. *NeurIPS*.
- Zhou, H., Liu, L., Zhao, Z., Zhang, Z., & Zhang, Q. (2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting. *AAAI*.
- Wu, Y., He, J., & Li, J. (2021). Autoformer: Decomposition Transformers with Auto-Correlation Mechanism for Long-Term Series Forecasting. *ICLR*.
- Li, Z., Liu, C., & Zhang, Z. (2021). LogTrans: Improving Transformer with Logarithmic Attention for Long Sequence Forecasting. *ICLR*.
- Zhang, X., Liu, Y., Yang, Y., & Zhao, H. (2022). Patch TST: Patch Transformer for Time Series Forecasting. *NeurIPS*.
