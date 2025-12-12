



## Model Fine-tuning Initialization (60GB GRAM Needed)
First download the dataset from https://www.kaggle.com/datasets/yousefsaeedian/ai-medical-chatbot
then reformat the dataset into .parquet

To start fine-tuning, put the .parquet dataset under the same directory as SFT.ipynb
Then, run SFT.ipynb script on a device that has >= 60 GRAM, the final cell will save the model trained params

To start evaluation via SARI, make sure model folder, test_sari.py and Qwen3-8B-SARI.ipynb are all under the same directory
Then, run Qwen3-8B-SARI.ipynb and the evaluation results will be shown in the console log