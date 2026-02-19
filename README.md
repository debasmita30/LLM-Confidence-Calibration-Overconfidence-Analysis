# LLM-Confidence-Calibration-Overconfidence-Analysis
A rigorous evaluation of calibration, overconfidence, and hallucination behavior in instruction-tuned Large Language Models (LLMs).
This project quantitatively measures and corrects miscalibration using logit-level analysis and post-hoc temperature scaling.

## 📌 Overview

Large Language Models often produce highly confident predictions—even when incorrect.
This project investigates:

How overconfident modern LLMs are

Whether self-reported confidence is reliable

How temperature scaling improves calibration

How calibration differs across model sizes

The study compares:

Mistral-7B-Instruct

Phi-2

using statistical evaluation metrics rather than prompt-based heuristics.

🎯 Objectives

Measure probabilistic calibration of LLMs

Quantify overconfident hallucinations

Apply post-hoc temperature scaling

Compare internal softmax confidence vs self-reported confidence

Evaluate calibration differences between large and smaller models

Provide statistically grounded evaluation using bootstrap confidence intervals

## 🧪 Experimental Setup
Dataset

BoolQ (Yes/No question answering)

500–1000 validation samples

Proper calibration/test split:

50% calibration set

50% evaluation set

Models Evaluated

mistralai/Mistral-7B-Instruct-v0.2 (4-bit quantized)

microsoft/phi-2

Confidence Extraction

Logits extracted from final token

Softmax probabilities computed for "yes" and "no"

Confidence = max probability

Post-hoc Calibration

Temperature scaling via grid search

Optimization based on minimizing Negative Log Likelihood (NLL)

Evaluation strictly performed on held-out test set

## 📊 Metrics Used

Accuracy

Expected Calibration Error (ECE)

Adaptive ECE (equal-frequency binning)

Brier Score

Negative Log Likelihood (NLL)

Overconfident Hallucination Rate

Bootstrap Confidence Interval for ECE reduction

## 🔍 Key Results
🔵 Mistral-7B-Instruct

Accuracy: 81.2%

Raw ECE: 0.1588

Calibrated ECE: 0.0603

Optimized Temperature: 6.89

Overconfident Hallucinations:

Before: 18.4%

After: 13.6%

ECE Reduction: ~62%

Observation:
Strong overconfidence. Requires large temperature correction.

🟢 Phi-2

Accuracy: 80.0%

Raw ECE: 0.0524

Calibrated ECE: 0.0322

Optimized Temperature: 1.35

Observation:
Better default calibration. Requires minimal correction.

## 🧠 Major Insights
1️⃣ Larger Models Can Be Less Calibrated

Mistral-7B is slightly more accurate but significantly more overconfident.

2️⃣ Overconfidence Scales With Model Sharpness

Optimized temperature of ~6.9 indicates extreme logit sharpness.

3️⃣ Calibration Improves Reliability Without Affecting Accuracy

Temperature scaling reduced ECE and overconfident hallucinations while keeping accuracy constant.

4️⃣ Self-Reported Confidence Is Unreliable

Weak correlation with softmax confidence (ρ ≈ 0.10)

Worse Brier score than softmax probabilities

Often saturated near 1.0

Conclusion: Prompt-based confidence is not a reliable uncertainty estimator.

## 📈 Reliability Diagrams

The plots below show calibration behavior before and after temperature scaling.

- Points below the diagonal indicate overconfidence.
- Temperature scaling shifts predictions closer to the ideal calibration line.

![Reliability Diagrams] reliability_diagrams.png

🔬 Statistical Validation

Bootstrap resampling (1000 iterations) was used to compute a 95% confidence interval for ECE reduction, ensuring statistical robustness of calibration improvements.

## 🛠️ Technologies Used

Python

PyTorch

Hugging Face Transformers

BitsAndBytes (4-bit quantization)

NumPy / Pandas

Matplotlib

SciPy

## 📂 Project Structure
LLM_Calibration_Study.ipynb
calibration_results.csv
README.md

🚀 Why This Project Matters

Most prompt engineering projects focus on prompt wording.
This project focuses on:

Logit-level evaluation

Statistical calibration analysis

Cross-model comparison

Quantitative uncertainty assessment

Systematic reliability improvement

It bridges prompt engineering with model evaluation and applied ML research.

## 🧩 Future Improvements

Adaptive binning with dynamic bin count

Confidence-aware decoding strategies

Calibration across multiple datasets

Comparison with larger models (e.g., Llama-3)

Selective prediction / abstention mechanisms

## 📜 License

For educational and research purposes.

👤 Author

[Debasmita Chatterjee]
Focus: LLM Evaluation, Calibration, and Applied AI Systems
