# Chest-X-ray-DL
> Our DL 289G Project


## Project Overall
Deep learning in Computer Vision is important in medical area. Our project is mainly working on disease prediction for patients using X-ray images with technologies, such as CNNs and RNNs. In this project, we will present a deep learning model, which takes in a sequence of the consecutive previous chest X-rays of patients, analyze the variation and difference across this sequence. For the feature extraction phase of the images, the model uses convolutional neural networks (CNN), such as DenseNet, MobileNet, and ResNet. Besides these, we also compare and analyze specifically the impact of LSTMs on these X-ray based on the extracted feature maps from experimental CNN models. In conclusion, throughout this project, we intend to present a single deep learning framework, which would take in more than one X-ray per patient for analysis and would intend to treat these X-rays as an image sequence which would be then used for predicting the disease label based on the differences observed within the regions present across each follow-up X-ray, and our goal is to identify how does follow-up X-ray images play a significant role in predicting the disease labels. 

The dataset we used is found here: https://www.kaggle.com/nih-chest-xrays/data

## Brief Script Description and Usage

### Preprocessing Scripts
> scripts that are used for data preproccessing and data cleaning
1. `Filter_and_Create_Sample_Sets.ipynb`
    1. Pick Patients who have at least 3 followups (indexing from 0)
    1. Create two different sample datasets based on view position and store datasets into CSVs
    1. Relevant CSV files for this script:
        1. `Data_Entry_2017.csv`
        1. `df_updated_view_postion.csv`
        1. `df_updated_finding_labels.csv`
        1. `df_PA.csv`
        1. `df_AP.csv`

1. `Preprocess_Analyze_Image_Datasets.ipynb`
    1. PA, PA images dataset processing
        1. Adding Full Paths and Some basic preprocessing
        1. Train, Test, Validation dataset creation
        1. Analyzing the samples for label distributions
    1. Saving preprocessed into arrays and store in pickle.
    1. Relevant files for this script:
        1. `df_PA.csv`
        1. `df_AP.csv`
        1. `added_paths_PA.csv`
        1. `added_paths_AP.csv`
        1. `PA_train.csv`
        1. `PA_test.csv`
        1. `PA_val.csv`
        1. `AP_train.csv`
        1. `AP_test.csv`
        1. `AP_val.csv`
        1. `AP_images.pkl`
        1. `PA_images.pkl`

1. `Process_NIH_Dataset_Details.ipynb`
    1. process NIH dataset details
    1. data analysis using data visualization
    1. Relevant files for this script:
        1. `BBox_List_2017.csv`
        1. `Data_Entry_2017.csv`

1. `Sample_Set_Images.ipynb`
    1. PA, AP Position manual Feature Extraction
    1. Relevant files for this script:
        1. `df_AP.csv`

1. `verify_files.py`
    1. check if files are correctly merged

### single_image_models scripts
> scripts that used for single image input models 
1. `AP_X_ray_images_baseline_dataprocessing_v2.ipynb` and `PA_X_ray_images_baseline_dataprocessing_v2.ipynb`
    1. For single image preprocessing, we added dataframes for AP or PA (from `df_pa.csv` and `df_ap.csv`), and then we linked images from google drive and then save them to `added_paths_ap.csv` and `added_paths_pa.csv`. We have split that datasets into three one with train, val, and test. We have then resized the images and saved as pickle files
    1. Relevant files for this script:
        1. `df_AP.csv`
        1. `added_paths_AP.csv`
        1. `train_AP.pkl`
        1. `val_AP.pkl`
        1. `test_AP.pkl`
        1. `df_PA.csv`
        1. `added_paths_PA.csv`
        1. `train_PA.pkl`
        1. `val_PA.pkl`
        1. `test_PA.pkl`


1. `Single_Xray_AP_results.ipynb` and `Single_Xray_PA_results.ipynb`
    1. storing and analyzing results for single AP and PA X-ray images
    1. Relevant files for this script:
        1. `added_paths_PA.csv`
        1. `added_paths_AP.csv`
        1. `train_df_DenseNet.csv`
        1. `valid_df_DenseNet.csv`
        1. `test_df_DenseNet.csv`

1. `APmodelling.py` and `PAmodelling.py`
    1. To compare DenseNet, ResNet, and MobileNet, we have tested our datasets on a simple CNN model which contained 5 layers, 1000 units, and kernel size of 7. The dropout rate was 40% and used softmax activation function. We have used Adam optimizer. Our CNN model will have 15 outputs. Loss function we used was categorical cross entropy, and we used accuracy metrics. After processing on the CNN, we saved our results on pickle files
    1. Relevant files for this script:
        1. `train.pkl`
        1. `val.pkl`
        1. `test.pkl`

### three_image_models scripts
> scripts that used for three images input models
1. `BaseModelScript.ipynb`
    1. Load images and get the outputs: X,y creation
    1. For both PA and AP
        1. Train, test, validate X,Y sets
        1. DenseNet modeling experiment with LSTM/without LSTM
    1. Relevant files for this script:
        1. `PA_images.pkl`
        1. `AP_images.pkl`
1. `DenseNetPAModellingFinal.ipynb` and `DenseNet_AP_Modeling.ipynb`
    1. DenseNet169 in-depth modeling experiment with LSTM/without LSTM on PA and AP
    1. DenseNet169 with LSTM/without LSTM result ROC analysis
    1. DenseNet169 with LSTM/without LSTM result Loss analysis
    1. DenseNet169 with LSTM/without LSTM result Accuracy analysis
    1. Relevant files for this script:
        1. `PA_train.csv`
        1. `PA_test.csv`
        1. `PA_val.csv`
        1. `AP_train.csv`
        1. `AP_test.csv`
        1. `AP_val.csv`
        1. `PA_images.pkl`
        1. `AP_images.pkl`
1. `Modeling_MobileNetV2_AP_.ipynb` and `Modeling_MobileNetV2_PA_.ipynb`
    1. MobileNetV2 in-depth modeling experiment with LSTM/without LSTM on PA and AP
    1. MobileNetV2 with LSTM/without LSTM result ROC analysis
    1. MobileNetV2 with LSTM/without LSTM result Loss analysis
    1. MobileNetV2 with LSTM/without LSTM result Accuracy analysis
    1. Relevant files for this script:
        1. `PA_train.csv`
        1. `PA_test.csv`
        1. `PA_val.csv`
        1. `AP_train.csv`
        1. `AP_test.csv`
        1. `AP_val.csv`
        1. `PA_images.pkl`
        1. `AP_images.pkl`
1. `Modeling_ResNetV2_AP_.ipynb` and `Modeling_ResNetV2_PA_.ipynb`
    1. ResNet50V2 in-depth modeling experiment with LSTM/without LSTM on PA and AP
    1. ResNet50V2 with LSTM/without LSTM result ROC analysis
    1. ResNet50V2 with LSTM/without LSTM result Loss analysis
    1. ResNet50V2 with LSTM/without LSTM result Accuracy analysis
    1. Relevant files for this script:
        1. `PA_train.csv`
        1. `PA_test.csv`
        1. `PA_val.csv`
        1. `AP_train.csv`
        1. `AP_test.csv`
        1. `AP_val.csv`
        1. `PA_images.pkl`
        1. `AP_images.pkl`
1. `Loss_Acc_Plots.ipynb`
    1. a summary version of Loss plots and Acc plots for DenseNet, MobileNetV2, ResNetV2 experiments on the architecture with/without LSTM

#### Applied Dependencies
1. Pandas
1. Numpy
1. Keras
1. Tensorflow
1. OS
1. CSV
1. Pickle
1. tqdm
1. Sklearn
1. Collections
1. PIL
1. Matplotlib
1. Seaborn
1. glob
1. CV2
1. Time
1. Google.colab

#### File Dependencies
> files stored in data_csv_files directory
1. `added_paths_AP.csv`
    contains the corresponding full file path for each AP datapoints' X-ray image on google drive
1. `added_paths_PA.csv`
    contains the corresponding full file path for each PA datapoints' X-ray image on google drive
1. `AP_test.csv`
    contains the test set for AP
1. `PA_test.csv`
    contains the test set for PA
1. `AP_val.csv`
    contains the validation set for AP
1. `PA_val.csv`
    contains the validation set for PA
1. `AP_train.csv`
    contains the training set for AP
1. `PA_train.csv`
    contains the training set for PA

