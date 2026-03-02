# LLM-Confidence-Calibration-Overconfidence-Analysis

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFCC4D?logo=huggingface&logoColor=black)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-150458?logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?logo=plotly&logoColor=white)
![Calibration](https://img.shields.io/badge/Model-Calibration-purple)
![ECE](https://img.shields.io/badge/Expected_Calibration_Error-Metric-green)
![Bootstrap](https://img.shields.io/badge/Bootstrap-Statistical_Validation-orange)
![Reliability](https://img.shields.io/badge/Reliability-Diagram-success)

A production-grade statistical evaluation framework for diagnosing and correcting overconfidence in instruction-tuned Large Language Models (LLMs).

This system performs logit-level confidence extraction, probabilistic calibration analysis, hallucination quantification, and post-hoc temperature scaling — ensuring deployment-grade reliability without affecting model accuracy.

🧠 Problem Statement

LLMs frequently produce highly confident predictions—even when incorrect.

In production environments, this leads to:

Overconfident hallucinations

Misleading decision support

Risk amplification in enterprise systems

Reduced trust in AI outputs

This project provides a mathematically grounded framework to measure and correct model miscalibration.

🎯 System Capabilities

✔ Logit-level confidence extraction
✔ Expected Calibration Error (ECE) measurement
✔ Adaptive binning calibration analysis
✔ Overconfident hallucination detection
✔ Post-hoc temperature scaling optimization
✔ Bootstrap statistical validation
✔ Cross-model calibration benchmarking

Designed for reliability analysis in real-world AI deployment settings.

🧪 Experimental Framework
Dataset

BoolQ (Yes/No QA)

500–1000 validation samples

Strict split:

50% calibration set

50% held-out evaluation set

No calibration leakage.

Models Evaluated

mistralai/Mistral-7B-Instruct-v0.2 (4-bit quantized)

microsoft/phi-2

⚙️ Core Methodology

1️⃣ Logit-Level Confidence Extraction

Extract final-token logits

Compute softmax over binary labels

Confidence = max probability

Compare against ground truth

2️⃣ Calibration Metrics

Accuracy

Expected Calibration Error (ECE)

Adaptive ECE (equal-frequency bins)

Brier Score

Negative Log Likelihood (NLL)

Overconfident Hallucination Rate

3️⃣ Post-Hoc Temperature Scaling

Optimization objective:

Minimize Negative Log Likelihood (NLL)

Method:

Grid search temperature values

Fit on calibration set

Evaluate on held-out test set

Ensures:

Improved probabilistic alignment

No accuracy distortion

4️⃣ Statistical Validation

1000-iteration bootstrap resampling

95% confidence interval for ECE reduction

Ensures reproducibility and statistical robustness

📊 Quantitative Results
🔵 Mistral-7B-Instruct

Accuracy: 81.2%

Raw ECE: 0.1588

Calibrated ECE: 0.0603

Optimized Temperature: 6.89

ECE Reduction: ~62%

Overconfident Hallucinations:

Before: 18.4%

After: 13.6%

Observation: Larger model exhibits extreme logit sharpness and strong overconfidence.

🟢 Phi-2

Accuracy: 80.0%

Raw ECE: 0.0524

Calibrated ECE: 0.0322

Optimized Temperature: 1.35

Observation: Smaller model demonstrates stronger inherent calibration.

📈 Reliability Diagnostics

Reliability diagrams reveal:

Systematic overconfidence (points below diagonal)

Post-scaling alignment toward ideal calibration

Reduced probability distortion

Temperature scaling shifts predictions closer to true empirical likelihood.

📊 Key Insights Demonstrated

✔ Larger models can be less calibrated
✔ Logit sharpness correlates with overconfidence
✔ Self-reported confidence is unreliable (ρ ≈ 0.10)
✔ Calibration reduces ECE without affecting accuracy
✔ Prompt-based confidence is not a valid uncertainty estimator

🏗️ Project Structure
LLM_Calibration_Study.ipynb
calibration_results.csv
README.md
🚀 Production Relevance

This system demonstrates:

Deployment-grade uncertainty estimation

Statistical model evaluation discipline

Hallucination risk quantification

Cross-model reliability benchmarking

Probabilistic correction without retraining

Applicable to:

Enterprise AI systems

Conversational AI evaluation

RLHF quality diagnostics

Model benchmarking pipelines

Human-in-the-loop AI validation

🔬 Research Extensions

Adaptive dynamic bin calibration

Selective prediction / abstention

Confidence-aware decoding

Multi-dataset calibration benchmarking

Frontier model comparison

📄 Executive Summary

This project provides a mathematically rigorous calibration framework for instruction-tuned LLMs. By integrating logit-level confidence extraction, statistical reliability diagnostics, and post-hoc temperature scaling, it improves probabilistic correctness without altering model accuracy.

Designed for researchers, AI evaluators, and engineers building trustworthy LLM systems.

👩‍💻 Author

Debasmita Chatterjee
LLM Evaluation | Calibration | Applied AI Systems
