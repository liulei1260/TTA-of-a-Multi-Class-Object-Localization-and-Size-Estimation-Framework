# An Open Set Recognition Framework of Multi-class Object Counting and Size Estimation for Bundle Design in Grocery Retailing
Online product bundle design by considering customers' consumption behaviour is an effective way to attract customers' attention and therefore increase the purchase probability. But without customized recommendations, the number of all available bundles increases exponentially as the attributes and corresponding levels increase. Besides, the traditional way of using the qualitative word in the questionnaire to collect customers' preferences cannot correctly represent such quantitate attributes and corresponding levels. In this paper, we designed a discrete choice experiment to collect customers' bundle choice preferences in which the attribute and levels are presented in images and adopt an orthogonal factorial design to formulate the choice set. A domain adaptation-based deep learning approach for multi-class object Localization and size estimation problems in an open set recognition setting is proposed and applied in our bundle choice experiment and model to automatically extract information on the product size and quantity from the real image. This approach is the first attempt to handle source-free domain adaptation on a heat map regression task. The results are achieved in one epoch of test-time optimization without altering the training process.

The dataset of extended MOCSE13 used to train and test the backbone model and the fullly testing time adapation model for multi-class object Localization and size estimation problems can be download from https://www.dropbox.com/sh/hi7tqltr76arhof/AAA2tVFKSqQiFfGAD9rx_WIPa?dl=0 
We could create customized data (new type of object) based on the requirements.

# How to use
1. Modify the data path and parameter settings as needed
2. Use the tranin.py file to train the backbone model, use test+eval_model.py to evaluate all checkpoints by using the test images under data folder of "traning classes of data". test.py is use to test a model of single checkpoint.  
3. test_tta.py and test_ttt.py are used to do the fullly testing time adapation for the backbone model by using the test images under data folder of "testing classes of data".

# Reference

# Contact
zixu.liu@soton.ac.uk  xi.yang01@xjtlu.edu.cn  
