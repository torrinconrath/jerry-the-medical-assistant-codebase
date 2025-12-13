## Dataset Processing
First download the dataset from https://www.kaggle.com/datasets/yousefsaeedian/ai-medical-chatbot

Look into DataOutputProcessing.ipynb:
Then you need to set up a huggingface account and put your token inside the notebook.
After that make sure you have the installed the necessary imports and run the file.

## Model Fine-tuning (60 GB VRAM NEEDED, WE USED GOOGLE COLAB PRO FOR THIS)
First reformat the subset dataset into .parquet. (Or use https://drive.google.com/drive/folders/108_8uL-6HTn1_Rf22fRdGFNzrxXGzDX2?usp=sharing to download pre-formatted dataset)

To start fine-tuning, put the .parquet dataset under the same directory as SFT.ipynb Then, run SFT.ipynb script on a device that has >= 60 GRAM, the final cell will save the model trained params (or https://drive.google.com/file/d/13xoplY0-SrNULChwuSAo988l828CaEEc/view?usp=sharing to directly download the fine-tuned model).

To start evaluation via SARI, make sure model folder, test_sari.py and Qwen3-8B-SARI.ipynb are all under the same directory.
Then, run Qwen3-8B-SARI.ipynb and the evaluation results will be shown in the console log.

## Application (8 GB VRAM NEEDED)

For running the application, you will need the Windows Subsystem for Linux (WSL) and a tunneling service (Ngrok) dev domain.

### Backend

Firstly setup a Python environment with all your dependencies needed for the script.
Place the server script in your Linux system, then configure your routing for the model files. Make sure you have converted your model files into AWQ (Just use autoAWQ and create a quick script for it) and update the MODEL_PATH Constant. If you don't have the time, you can also just change awq_marlin to bitsandbytes and run it that way but it will be slower and require at least 12 GB of VRAM.
Then input your ngrok token and dev domain into the environment file in the WSL subsystem and make sure it matches these names: "NGROK_AUTHTOKEN", "DEV_DOMAIN".

Lastly, you can run the server script.

### Frontend

The frontend should be simplier. Make sure you have node installed and cd into the files and run npm ci to install all the dependencies.
Then, just add your dev domain to a created env file that matches "VITE_SFT_MODEL_ENDPOINT".

Lastly, you can run the frontend using npm run dev.
