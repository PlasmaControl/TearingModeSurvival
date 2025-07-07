# TearingModeSurvival
A tearing mode prediction model based on the auton-survival deep survival machine. Further details on results using this repository can be found in our publication **Interpreting AI for Fusion: an application to Plasma Profile Analysis for Tearing Mode Stability**: https://arxiv.org/abs/2502.20294

For simply trying out the model:
- The models, data and configs are located in the Princeton University clusters such as stellar, in /projects/EKOLEMEN/survival_tm_2/ in folders of their name
- Use tm_model_simple_analysis.ipynb for trying out predictions on any shot in the database

For more advanced use, the repository includes the following features:
1) Creating and formatting the database from the data-fetching repo
2) Training the model
3) Basic model analysis and inference
4) Shap analysis

**Creating the database**
Use data_processing_main.ipynb to create a DSM-compatible database from TM event labels and plasma data. The data is extracted from DIII-D using the PlasmaControl/data-fetching repository, and the tearing mode event labels are created using the criteria outlined in the publication. 

**Training the model**
- For a simple model training, run train_tm_model.py editing model.cfg to use the desired training databases and hyperparameters. 
- For running a batch script on the Princeton Stellar and Della clusters, use launch_survival_training.py, which will automatically submit a batch job using train_tm_model.py. 
- For hyperparameter tuning using ray tube, run hyperparameter_tuner.py or launch_hyperparameter_tuning.py for the batch submission. These will read from hyperparam_model.cfg

**Basic model analysis**
Use tm_model_simple_analysis.ipynb for analysing training progress, tearing mode predictions and creating ROC curves.

**Shap analysis**
Use shap_analysis.ipynb to run shapley analysis of the tearing mode prediction model. This script includes individual profile analysis as well as database-wise scans using beeswarm plots. 
