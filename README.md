# Leveraging Large Language Models for Marketing Analytics

This repository contains the code and resources for the paper **"Leveraging Large Language Models for Marketing Analytics"**. The study explores how Large Language Models (LLMs) can be used to analyze consumer feedback on Lavazza products, transforming unstructured data into actionable marketing insights. The goal is to help businesses understand customer sentiment, improve product launches, and make data-driven decisions.

---

## Abstract

Analyzing social media comments and reviews is vital for companies aiming to evaluate the true impact of product launches. This paper introduces an **AI-driven pipeline** designed to convert online feedback about Lavazza products into actionable marketing insights using a multi-level classification framework powered by **Large Language Models (LLMs)**. The proposed architecture processes feedback through four key stages:
1. **Product relevance classification**
2. **Multi-topic sentiment detection**
3. **Overall sentiment analysis**
4. **Star rating prediction**

The pipeline was tested using both **general-purpose LLMs** (Gemma-2, Llama-3.1, Mistral) and **specialized BERT-based models**, achieving high accuracy across all tasks. The results demonstrated **82.6% accuracy** in product relevance classification, **89.9% F1 score** for positive sentiment detection, **87.7% overall sentiment accuracy**, and **69.2% star rating accuracy**.

---

## Introduction

In today’s digital era, understanding consumer responses to product launches is essential for business success. Social media platforms and review websites generate a wealth of user-generated feedback, offering valuable insights that can shape marketing strategies. However, the high volume and unstructured nature of this data make manual analysis both time-consuming and impractical.

This research addresses the challenge by developing an **AI-driven pipeline** that leverages **Large Language Models (LLMs)** to analyze consumer feedback. The pipeline employs a **multi-level classification framework** to process social media comments and product reviews systematically. It begins with **product relevance classification** to filter out unrelated comments, followed by **topic extraction with sentiment tagging**, **overall sentiment analysis**, and **star rating prediction**. The outputs of these classification tasks are aggregated into **Key Performance Indicators (KPIs)**, which provide actionable insights for marketing teams.

---

## Methodology

### Data Collection and Preprocessing

The dataset used in this study includes:
- **2,930 reviews** provided by Lavazza.
- **577 social media comments** collected from Instagram and YouTube.

The data was preprocessed through the following steps:
- **Data Cleaning**: Removal of duplicate entries and reviews with null values.
- **Text Normalization**: Standardizing text by removing special characters and extra spaces.
- **Topic Identification**: Categorizing 103 unique product aspects into 9 meaningful categories (e.g., Coffee Quality, Machine Performance, User Experience).
- **Translation**: Non-English reviews were translated into English for consistency.

### Multi-Level Classification Architecture

The pipeline processes feedback through a series of hierarchical classification tasks:

1. **Product-Relevant Comment Classification**: Filters out comments that are not related to the product.
2. **Multi-Topic Sentiment Classification**: Identifies specific product aspects and assigns sentiment to each.
3. **Overall Sentiment Classification**: Determines the general sentiment of the feedback.
4. **Star Rating Prediction**: Predicts star ratings based on the sentiment and tone of the reviews.

### Models

Two categories of models were used in this study:
- **General-Purpose Models**: Gemma-2, Llama-3.1, and Mistral. These models are versatile and can handle a wide range of NLP tasks.
- **Specialized Models**: Fine-tuned BERT-based models for specific tasks like sentiment analysis and star rating prediction.

---

## Experiments and Results

The experiments evaluated the performance of the models under **zero-shot** and **few-shot** configurations. Below are the key findings, with the highest values in each column highlighted in **bold**.

### 1. Product-Relevant Comment Classification

This task aims to filter out irrelevant comments. The table below shows the performance of different models:

| Model            | Accuracy | Precision | Recall | F1-Score | Inference Time (s) |
|------------------|----------|-----------|--------|----------|--------------------|
| Gemma 0-shot     | 0.758    | 0.738     | 0.896  | 0.809    | 3.40               |
| Gemma Few-shot   | 0.814    | 0.783     | 0.933  | 0.851    | 3.63               |
| Llama 0-shot     | 0.763    | 0.737     | 0.911  | 0.815    | **2.77**           |
| Llama Few-shot   | **0.826**| **0.794** | **0.941**| **0.861**| 2.93               |
| Mistral 0-shot   | 0.758    | 0.747     | 0.874  | 0.805    | 3.25               |
| Mistral Few-shot | 0.797    | 0.764     | 0.933  | 0.840    | 3.53               |

**Best Model**: **Llama Few-shot** achieved the highest accuracy (**82.6%**) and F1-score (**86.1%**).

---

### 2. Multi-Topic Sentiment Classification

This task identifies specific product aspects and assigns sentiment to each. The table below shows the performance of different models:

| Model            | F1 (Neg.) | F1 (Pos.) | Precision (Neg.) | Precision (Pos.) | Recall (Neg.) | Recall (Pos.) | Inference Time (s) |
|------------------|-----------|-----------|------------------|------------------|---------------|---------------|--------------------|
| Gemma 0-shot     | 0.895     | 0.732     | 0.901            | 0.834            | 0.900         | 0.705         | 12.10              |
| Gemma Few-shot   | **0.899** | 0.732     | **0.905**        | **0.844**        | **0.904**     | 0.694         | 12.82              |
| Llama 0-shot     | 0.880     | **0.745** | 0.901            | 0.795            | 0.879         | 0.718         | **8.27**           |
| Llama Few-shot   | 0.878     | 0.730     | 0.895            | 0.811            | 0.874         | 0.694         | 9.55               |
| Mistral 0-shot   | 0.873     | 0.738     | 0.885            | 0.761            | 0.873         | **0.735**     | 11.12              |
| Mistral Few-shot | 0.880     | 0.715     | **0.905**        | 0.789            | 0.873         | 0.679         | 20.96              |

**Best Model**: **Gemma Few-shot** performed best, with an F1 score of **89.9%** for negative sentiment.

---

### 3. Overall Sentiment Classification

This task determines the general sentiment of the feedback (positive, negative, or neutral). The table below shows the performance of different models:

| Model            | Accuracy | Precision | Recall | F1-Score | Inference Time (s) |
|------------------|----------|-----------|--------|----------|--------------------|
| Gemma 0-shot     | 0.869    | 0.878     | 0.869  | 0.866    | 3.75               |
| Gemma Few-shot   | 0.852    | 0.856     | 0.852  | 0.849    | 3.95               |
| Llama 0-shot     | 0.818    | 0.818     | 0.818  | 0.815    | 2.76               |
| Llama Few-shot   | 0.814    | 0.825     | 0.814  | 0.812    | 2.96               |
| Mistral 0-shot   | **0.877**| **0.880** | **0.877**| **0.876**| 4.75               |
| Mistral Few-shot | 0.860    | 0.860     | 0.860  | 0.859    | 3.71               |
| DB-sentimet      | 0.496    | 0.436     | 0.496  | 0.388    | **6.52×10⁻³**      |
| TR-sentimet      | 0.682    | 0.751     | 0.682  | 0.680    | 6.77×10⁻³          |
| TRI-sentimet     | 0.831    | 0.837     | 0.831  | 0.829    | 8.41×10⁻³          |

**Best Model**: **Mistral 0-shot** achieved the highest accuracy (**87.7%**) and F1-score (**87.6%**).

---

### 4. Star Rating Prediction

This task predicts star ratings (1 to 5) based on the sentiment and tone of the reviews. The table below shows the performance of different models:

| Model            | Accuracy | Precision | Recall | F1-Score | Off-by-One Accuracy | Inference Time (s) |
|------------------|----------|-----------|--------|----------|---------------------|--------------------|
| Gemma 0-shot     | 0.696    | 0.763     | 0.596  | 0.650    | 0.908               | 8.07               |
| Gemma Few-shot   | 0.592    | 0.765     | 0.592  | 0.647    | 0.908               | 8.17               |
| Llama 0-shot     | 0.682    | 0.764     | 0.682  | 0.713    | 0.948               | 5.73               |
| Llama Few-shot   | 0.682    | **0.777** | 0.682  | 0.715    | 0.942               | 5.92               |
| Mistral 0-shot   | 0.498    | 0.741     | 0.498  | 0.567    | 0.912               | 6.12               |
| Mistral Few-shot | 0.434    | 0.732     | 0.434  | 0.510    | 0.902               | 6.45               |
| BBM-rating       | **0.692**| 0.757     | **0.692**| **0.718**| **0.955**           | 0.158              |
| MLA-rating       | 0.552    | 0.706     | 0.552  | 0.606    | 0.845               | **10.45×10⁻³**     |

**Best Model**: **BBM-rating** achieved the highest accuracy (**69.2%**) and off-by-one accuracy (**95.5%**).

---

## Conclusion

This study demonstrates the potential of **Large Language Models (LLMs)** in extracting actionable insights from consumer feedback. General-purpose models like **Gemma Few-shot** excel in product relevance and multi-topic sentiment analysis, while specialized models like **BBM-rating** offer faster inference times for star rating prediction.

Future work could explore advanced prompting methods, such as **chain-of-thought reasoning**, to further enhance model accuracy and enable more complex analysis of customer feedback.

---

## References

For more details, refer to the original paper and the following resources:

- [Gemma-2](https://arxiv.org/abs/2408.00118)
- [Llama-3.1](https://arxiv.org/abs/2407.21783)
- [Mistral](https://arxiv.org/abs/2310.06825)
- [BERT-based Models](https://huggingface.co/models)

You can also find the full report in this repository: [ADSP_P14_Final_Report.pdf](ADSP_P14_Final_Report.pdf).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.