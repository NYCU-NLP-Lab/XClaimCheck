# 🪄XClaimCheck🪄

XClaimCheck is a cross-domain fact-checking dataset introducted in [MAGIC: Multi-Argument Generation with Self-Refinement for Domain Generalization in Automatic Fact-Checking]().
Each claim is annotated with domain information and further categorized into five major domains, establishing a new benchmark for cross-domain fact-checking. 

## Introduction

This dataset is based on a previous work: [WatClaimCheck](https://github.com/nxii/WatClaimCheck)
We extend WatClaimCheck to study cross-domain fact-checking.
Given the necessity of associating claims with topic labels, we chose [PolitiFact](https://www.politifact.com/) as our data source due to its stable and diverse collection of domains.
We thus collected a total of 15,867 claims from PolitiFact in WatClaimCheck to construct XClaimCheck.
As the claims in WatClaimCheck were not tagged with topic labels, we obtained the topic information from PolitiFact. 
We identified a total of 26 representative topics, each containing a substantial amount of data suitable for our fact-checking task.
Note that certain claims, due to their multifaceted nature, pertain to multiple topics.
In such cases, we assigned these claims and their associated data to all relevant topics, resulting in a total of 16,177 instances.
Then we organized these 26 topics into five distinct domains.
Each topic was grouped with other relevant topics whenever possible. 
The five domains in XClaimCheck are ``Public Policy and Finance``, ``Political Issues``, ``Legal and Regulatory Affairs``, ``Infrastructure and Services``, and ``Global Affairs and Security``, respectively.
For more details, please refer to the paper.

In this repo, we provide a direct access to XClaimCheck dataset, and code to construct the dataset.
We use WatClaimCheck's claim index refering to each claim, and metadata like premise articles and review articles can also be accessed in WatClaimCheck dataset using the index.
As a result, you have to request the access to WatClaimCheck dataset to use our dataset.

## Usage

### Script

Put WatClaimCheck dataset under this repo's folder: `/`.
Then you can simply use the script we provided as `script/script.py` by running:
```python
python script.py
```
This script would automatically install required packages and build up XClaimCheck dataset.
However, you can still execute each step by yourself.

### Requirements
- Requests

### Constructing dataset

**Collect dataset** 

When executing `collect_dataset.py`, it helps you crawl topic information of claims from PolitiFact.
You can specify the path of WatClaimCheck dataset, default is root `/`.
After Crawling, only claims with 26 selected topics would be retained.
It will eventually output a json file documenting each claim's index and corresponding url and topic information.
You can use these information without proceeding to the next step, based on your need.

**Split dataset** 

If you want to follow our approach building five domains dataset, run `split_dataset.py`.
It splits the collected dataset into 26 topics grouped in domains, and further divided into training, validation and test sets at a ratio of 6\:2\:2.
The file structure would look like this:
```bash
├── XClaimCheck_dataset
│   ├── Domain_1
│   │   │── Economy
│   │   │   ├── training.json
│   │   │   ├── validation.json
│   │   │   └── test.json
│   │   └── Taxes
│   ├── Domain_2
│   │   ├── Elections
│   │   └── Jobs
```
