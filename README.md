1.python2,tensorboard, pytorch-0.4.0

2.put pascal-voc2012 dataset in data
data
----gt (labels)
----img (images)
----train.txt (train name)
----val.txt  (val name)
(if you want to train your datasets, you can put the datasets as i showed above.)

3.python train.py 

4.python multi_scale_test.py

I didn't test the model 'my_deeplabv3_mit.py' in voc2012, and you can modify the model as long as you like.
But another model made by me can achieve 83.6 miou in pascal-voc2012,the result is http://host.robots.ox.ac.uk:8080/anonymous/TOEAAY.html.
we will upload the model soon.
this code is to help you to finish your project. you can use it in a semantic segmentation project.
