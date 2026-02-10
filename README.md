# AudioLeaks

This repository contains the artifacts for the paper **AudioLeaks**.

To support reproducibility while respecting privacy and ethical considerations, we provide three compressed data packages as described below.

## Repository Overview

We release the following three packages:

1. **ExperimentalResults**  
   This package contains the data sources for all experimental scenarios presented in the paper, including processed features and trained models.  
   For each scenario, running `python run.py` reproduces the structured results reported in the paper.

2. **SourceCode**  
   This package provides the source code for the technical approach described in the paper, including data processing pipelines.  
   By replacing the dataset with the corresponding scenario dataset and executing `python run.py`, users can reproduce the data processing results reported in the paper.

3. **Datasets**  
   This package contains the datasets associated with each scenario evaluated in the paper.

---

## ExperimentalResults Package

Below we provide a detailed description of the **ExperimentalResults** package.  
The directory structure is organized according to the paper sections and scenarios.

```
ExperimentalResults/
├──requirements.txt
├──Result.xlsx
├── 5 Scenario 1 Inferring Application Contexts
│   ├── 5.1 Identifying Active XR Applications
│   │   ├── classifier_resnet18.pth
│   │   ├── run.py
│   │   └── tmp_mel_split
│   └── 5.2 Revealing In-Application State
│       ├── classifier_resnet18.pth
│       ├── run.py
│       └── tmp_mel_split
├── 6 Scenario 2 Inferring XR Spatial Position
│   ├── 6.1 Virtual space
│   │   ├── classifier_resnet18.pth
│   │   ├── run.py
│   │   └── tmp_mel_split
│   └── 6.2 Physical space
│       ├── classifier_resnet18.pth
│       ├── run.py
│       └── tmp_mel_split
├── 7 Scenario 3 Inferring XR Virtual Meetings
│   └── 7.1 Game chat
│       ├── classifier_resnet18.pth
│       ├── run.py
│       └── tmp_mel_split
├── 8 Practical Impact Factors and Generalization
│   ├── 8.1 Practical Impact Factors
│   │   ├── (1) Impact of Ambient Noise
│   │   ├── (2) Impact of Playback Volume
│   │   ├── (3) Impact of User Movement
│   │   ├── (4) Impact of Different Users
│   │   ├── (5) Impact of Cross-Session
│   │   ├── classifier_resnet18.pth
│   │   └── run.py
│   └── 8.2 Generality Across XR Platforms
│       ├── htc
│       │   ├── classifier_resnet18.pth
│       │   ├── run.py
│       │   └── tmp_mel_split
│       ├── pico4
│       │   ├── classifier_resnet18.pth
│       │   ├── run.py
│       │   └── tmp_mel_split
│       ├── quest2
│       │   ├── classifier_resnet18.pth
│       │   ├── run.py
│       │   └── tmp_mel_split
│       └── quest3
│           ├── classifier_resnet18.pth
│           ├── run.py
│           └── tmp_mel_split
└── 9 Defenses
    ├── classifier_resnet18.pth
    ├── run.py
    └── tmp_mel_split
```

**Step 1.** In the first-level directory, the `requirements.txt` file lists all the libraries required for this folder. To install the dependencies, execute the following command (My Python version 3.9.11):
> `pip install -r requirements.txt `

**Step 2.** The `Result.xlsx` file provides the data presented in each scenario of the paper, serving as a reference outline for readers. 

**Step 3.** For each scenario (paper section), the folder contains 'tmp_mel_split', which is the corresponding dataset, "classifier_resnet18.pth" is the trained model. In addition, running these  `run.py`  allows you to obtain the results, consistent with those provided in `Result.xlsx` and in the paper. The corresponding commands are as follows:
> `python run.py`


**If the paper is accepted, we plan to release a more comprehensive artifact, including additional datasets and code. If you encounter any difficulties, please don't hesitate to reach out for assistance. Thank you sincerely for your interest, time and patience.**