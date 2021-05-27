[![Build Status](https://travis-ci.com/MDAIceland/WaterSecurity.svg?branch=master)](https://travis-ci.com/MDAIceland/WaterSecurity)

# Water Security

### About the project
The current climate change scenario predicts that almost half of the world's population will live in areas of high water stress by 2050 with limited access to fresh clean water. Governments, national, and international institutions, as well as water management companies, are looking for solutions that can address this growing global water demand. Cities are encouraged to take action on water security, to build resilience to water scarcity and manage this finite resource for the future. 

Based on financial, educational, environmental and demographical data, this project aims to display and predict the water security risks around the world. In order to do so, a Regression Machine Learning pipeline is deployed per risk category (e.g. risk of higher water prices or risk of declining water quality), and a forecast of that risk's severity is made. The regression model is based on engineered and selected features from the data mentioned above.

### Documentation
The documentation of the project can be found [here](https://mdaiceland.github.io/WaterSecurity/).

### Development and Local Deployment

Installing:
`pip install -r requirements.txt`

Running the Web App:
`python run.py`

Generating Documentation:
`python generate_documentation.py`
