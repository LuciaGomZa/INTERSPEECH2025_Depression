# Depression Detection
This repository contains the code for the INTERSPEECH2025 paper: "Speech and Text Foundation Models for Depression Detection: Cross-Task and Cross-Language Evaluation"

**Abstract:**

Automated depression detection is gaining attention due to its potential to improve psychiatric care. This study compares the performance of foundation models (FMs) in two datasets: an extended Distress Analysis Interview Corpus (DAIC+) in English, and Depressive Indicators during Casual Talks (DEPTALK) dataset in Spanish. HuBERT models and their fine-tuned versions for emotion recognition (ER) are used for speech. RoBERTa models and their ER variants are applied for text. Representations from FMs are grouped into context windows and processed by a Gated Recurrent Unit. Early fusion is used for multimodal analysis. Speech models perform similarly across datasets (F1$\approx$0.60). Text models perform better on DAIC+ than on DEPTALK (F1=0.70 vs 0.45). Multimodal models using FMs fine-tuned for ER perform best for both (F1=0.75 in DAIC+, 0.69 in DEPTALK), showing effectiveness across tasks and languages. Fairness evaluation reveals gender bias, which motivates future research on its alleviation. 

![Project screenshot](method.pdf)

## Files and folders
* data: description of the datasets used, and instructions on how to obtain them.
* codes: folder with the following...
  * 1_Automatic Speech Recognition (ASR): ...
<!-- * requirements.txt: required packages to be installed. -->

## Citation
If you find this work helpful, please cite our work as:

<!--
Gómez-Zaragozá, L., Marín-Morales, J., Alcañiz, M., Soleymani, M. (2025) Speech and Text Foundation Models for Depression Detection: Cross-Task and Cross-Language Evaluation. Proc. INTERSPEECH 2025, XXXX-XXXX, doi: 

```
 @inproceedings{gomezzaragoza23_interspeech,
  author={Lucía Gómez-Zaragozá and Simone Wills and Cristian Tejedor-Garcia and Javier Marín-Morales and Mariano Alcañiz and Helmer Strik},
  title={{Alzheimer Disease Classification through ASR-based Transcriptions: Exploring the Impact of Punctuation and Pauses}},
  year=2023,
  booktitle={Proc. INTERSPEECH 2023},
  pages={2403--2407},
  doi={10.21437/Interspeech.2023-1734}
}
```

-->
