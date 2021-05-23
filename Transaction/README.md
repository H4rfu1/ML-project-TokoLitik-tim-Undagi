1. This directory contains .ipnyb programs and json file in order to predict monthly sales of TokoLitik project. Yet, it's still under development. Programs flow explanation:
(a) Run dataset_preparation.ipynb program to produce required dataset for our model
(b) Run lstm_model.ipynb program to export the model
(c) Run predict.ipynb program to predict next 6 monthly sales

2. I plan to do the following in the next commit:
- Predict for at least 3 to 6 next monthly sales (outside the test set)
- Convert this program into TensorflowJS
