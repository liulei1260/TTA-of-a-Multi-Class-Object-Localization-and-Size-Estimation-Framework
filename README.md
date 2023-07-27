# Test-Time Adaptation of a Multi-Class Object Localization and Size Estimation Framework in the Application of Online Grocery Shopping
Most of the current online grocery platforms use a unified image for each designed bundle product to show roughly product information such as price and total weight. Lacking detailed size, type and weight information for every single item in each of the bundle boxes may decrease the customer purchase rate and satisfaction, especially for the fruit/vegetable bundle which is different from industrially manufactured products which are almost identical. This is usually constrained by the fact that providing such individual object information for the product (or bundle) images by hand annotation whilst packing is time and effort-consuming considering the huge number of different kinds of fruits \& vegetables sold by the grocery market under different seasons and places. To overcome the limitation, a domain adaptation-based deep learning approach for multi-class object counting and size estimation in an open set recognition (OSR) setting is proposed to automatically extract information on the product size and quantity from the image. This is the first attempt at source-free domain adaptation for a heatmap regression task. The approach achieves accurate predictions in one epoch of test-time optimization without altering the training process. Several experiments are implemented to evaluate the prediction accuracy in tasks of object localization, size estimation, and counting problems in an OSR setting. A synthetic dataset created by Unity 3D including 13 different fruits and vegetables is used to train and test our approach. This dataset could be customized and further used as a benchmark or testbed for future studies.

The dataset of extended MOCSE13 used to train and test the backbone model and the fully testing time adaptation model for multi-class object Localization and size estimation problems can be downloaded from https://www.dropbox.com/sh/hi7tqltr76arhof/AAA2tVFKSqQiFfGAD9rx_WIPa?dl=0 
We could create customized data (a new type of object) based on the requirements.

# How to use
1. Modify the data path and parameter settings as needed
2. Use the tranin.py file to train the backbone model, and use test+eval_model.py to evaluate all checkpoints by using the test images under the data folder of "training classes of data". test.py is used to test a model of the single checkpoint.  
3. test_tta.py and test_ttt.py are used to do the fullly testing time adaptation for the backbone model by using the test images under the data folder of "testing classes of data".

# Reference

# Contact
zixu.liu@soton.ac.uk  xi.yang01@xjtlu.edu.cn huan.yu@soton.ac.uk
