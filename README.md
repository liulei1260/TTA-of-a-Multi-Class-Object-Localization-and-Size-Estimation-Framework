# Test-Time Adaptation of a Multi-Class Object Localization and Size Estimation Framework

This paper focuses on exploring deep learning techniques to automate the localization, counting, and size estimation of various fruits or vegetables. Considering the huge number of different categories of fruits/vegetables that exist in the real world, achieving the localization, counting, and size estimation tasks in an Open Set Recognition (OSR) environment, i.e., for novel classes unseen during the training process, presents a challenge. This work proposes a test-time domain adaptation approach based on deep learning for multi-class object localization and size estimation in an OSR environment.

A new benchmark dataset (which includes synthetic and real image data) is created and collected to train, test and evaluate the approaches mentioned above for the open set object counting, localization, and size estimation problems. It could be customized and used as a benchmark or testbed for future studies.  

The dataset of E-MOCSE13 used to train and test the backbone model and the fully testing time adaptation model for multi-class object Localization and size estimation problems can be downloaded from https://www.dropbox.com/sh/hi7tqltr76arhof/AAA2tVFKSqQiFfGAD9rx_WIPa?dl=0 
We could create customized data (a new type of object) based on the requirements.

# How to use
1. Modify the data path and parameter settings as needed
2. Use the tranin.py file to train the backbone model, and use test+eval_model.py to evaluate all checkpoints by using the test images under the data folder of "training classes of data". test.py is used to test a model of the single checkpoint.  
3. test_tta.py and test_ttt.py are used to do the full testing time adaptation for the backbone model by using the test images under the data folder of "testing classes of data".

# Reference

# Contact
zixu.liu@soton.ac.uk  xi.yang01@xjtlu.edu.cn q.wu@tue.nl huan.yu@soton.ac.uk
