# Photo-z-ML-Estimator
This repository contains the source code for a machine learning pipeline for astronomical object classification and photo-z estimation.
<br>
### Model coefficient files:
- star_class.pt: the star photo-z classification ANN
- star_final.pt: the star photo-z regression ANN (after filter)
- gal_final.pt: the galaxy photo-z regression ANN (after filter)
- qso_final.pt: the quasar photo-z regression ANN (after filter)

The .zip file of the datasets was too large to upload to this repository, so the SQL_query.txt provides the link to the database with the SQL query template used to obtain the data. The query format, as well as the inspiration behind this project can be found on this SDSS DR14 page: https://skyserver.sdss.org/dr14/en/proj/advanced/quasars/query.aspx . 
<br>
Using the pre-trained weights in the .pt files, the models can be used as is with the structure contained in ANN_models.ipynb. This notebook also contains the training and validation processes for all ANN models. <br>
*** The code for training the ANNs was directly taken from Dr. Hashemi's DATA 534 MNIST example code, with alterations for this data application. *** The main changes to this code include: switching the type of source from image data to tabular data, reducing number of parameters to compensate for different numbers of features, the data pre-processing stage in its entirety, the inclusion of code for regression and classification applications, and the implementation of Dataset and DataLoader on non-torch native datasets. 

### Random Forest Models 
The RF_models.ipynb notebook contains the steps necessary to training and using a small RF classifier for object classification and galaxy/quasar anomaly detects. The data for these can be acquired from the SQL_query.txt file. 
<br>
*** The definition of the function that created the RF model from sci-kit learn was taken from Dr. Crosby's source code for the DATA 507 class. *** The main changes to this code include: switching model type from SVM to RF, the inclusion of a validation set, addition of confusion matrix, the entirety of data labelling and pre-processing.
