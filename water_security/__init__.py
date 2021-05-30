"""
## Water Security - Iceland
The current climate change scenario predicts that almost half of the world’s population will live in areas of high water stress by 2050 with limited access to fresh clean water. Governments, national, and international institutions, as well as water management companies, are looking for solutions that can address this growing global water demand. Cities are encouraged to take action on water security, to build resilience to water scarcity and manage this finite resource for the future.

Based on financial, educational, environmental and demographical data, this project aims to display and predict the water security risks around the world. In order to do so, a Regression Machine Learning pipeline is deployed per risk category (e.g. risk of higher water prices or risk of declining water quality), and a forecast of that risk's severity is made. The regression model is based on engineered and selected features from the data mentioned above.

**In order to get the final dataset from the preprocessing the following notebooks have to be executed in the given order:**

1. [prep_hdro_v2.ipynb](notebooks/prep_hdro_v2.html) 
2. [combine_unlabeled.ipynb](notebooks/combine_unlabeled.html)
3. [Dataset Normalization and Imputation.ipynb](notebooks/Dataset Normalization and Imputation.html)
4. [Cities Test Set Processing.ipynb](notebooks/Cities Test Set Processing.html)
5. [Merge Unlabeled to Labeled.ipynb](notebooks/Merge Unlabeled to Labeled.html)
6. [CitiesPopulationDensity.ipynb](notebooks/CitiesPopulationDensity.html)
"""
