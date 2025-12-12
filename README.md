



## Model Fine-tuning Initialization (60GB GRAM Needed)
First download the dataset from https://www.kaggle.com/datasets/yousefsaeedian/ai-medical-chatbot then reformat the dataset into .parquet (Or use https://drive.google.com/drive/folders/108_8uL-6HTn1_Rf22fRdGFNzrxXGzDX2?usp=sharing to download pre-formatted dataset)

To start fine-tuning, put the .parquet dataset under the same directory as SFT.ipynb
Then, run SFT.ipynb script on a device that has >= 60 GRAM, the final cell will save the model trained params (or https://drive.google.com/file/d/13xoplY0-SrNULChwuSAo988l828CaEEc/view?usp=sharing to directly download the fine-tuned model)

To start evaluation via SARI, make sure model folder, test_sari.py and Qwen3-8B-SARI.ipynb are all under the same directory
Then, run Qwen3-8B-SARI.ipynb and the evaluation results will be shown in the console log