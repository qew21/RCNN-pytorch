# RCNN 
Rich feature hierarchies for accurate object detection and semantic segmentation   

# 工程内容
这个程序是在Liu-Yicheng的TensorFlow版R-CNN基础上移植到Pytorch的内容

# 开发环境  
windows10 + python3.8 + pytorch1.7 + cv2 + scikit-learn    

# 数据集
采用17flowers据集, 官网下载：http://www.robots.ox.ac.uk/~vgg/data/flowers/17/  

# 程序说明   
1、config.py---网络定义、训练与数据处理所需要用到的参数      
2、Networks.py---用于定义Alexnet_Net模型、SVM模型、边框回归模型   
4、process_data.py---用于对训练数据集与微调数据集进行处理（选择性搜索、数据存取等）    
5、train_and_test.py---用于各类模型的训练与测试、主函数     
6、selectivesearch.py---选择性搜索源码       


# 文件说明   
1、train_list.txt---预训练数据，数据在17flowers文件夹中         
2、fine_tune_list.txt---微调数据2flowers文件夹中       
3、通过RCNN后的区域划分                   
![RCNN_1](https://github.com/qew21/RCNN-pytorch/blob/master/result/Figure_4-1.png)    
　　　
4、通过SVM与边框回归之后的最终结果      
![RCNN_2](https://github.com/qew21/RCNN-pytorch/blob/master/result/Figure_4-2.png)                     


# 论文细节补充：
1.finturn过程：
      
  计算每个region proposal与人工标注的框的IoU，IoU重叠阈值设为0.5，大于这个阈值的作为正样本，其他作     
　为负样本。然后在训练的每一次迭代中都使用32个正样本（包括所有类别）和96个背景样本组成的128张图片的batch    
　进行训练（正样本图片太少了）      

2.SVM训练过程：
          
　对每个类都训练一个线性的SVM分类器，训练SVM需要正负样本文件，这里的正样本就是ground-truth框中的图像作    
　为正样本，完全不包含的region proposal应该是负样本，但是对于部分包含某一类物体的region proposal该如  
　何训练作者同样是使用IoU阈值的方法，这次的阈值为0.3，计算每一个region proposal与标准框的IoU，小于0.3   
　的作为负样本，其他的全都丢弃。由于训练样本比较大，作者使用了standard hard negative mining method   
　（具体怎么弄的不清楚）来训练分类器。作者在补充材料中讨论了为什么fine-tuning和训练SVM时所用的正负样本   
　标准不一样，以及为什么不直接用卷积神经网络的输出来分类而要单独训练SVM来分类，作者提到，刚开始时只是用了   
　ImageNet预训练了CNN，并用提取的特征训练了SVMs，此时用正负样本标记方法就是前面所述的0.3,后来刚开始使用   
　fine-tuning时，使用了这个方法但是发现结果很差，于是通过调试选择了0.5这个方法，作者认为这样可以加大样本   
　的数量，从而避免过拟合。然而，IoU大于0.5就作为正样本会导致网络定位准确度的下降，故使用了SVM来做检测，全    
　部使用ground-truth样本作为正样本，且使用非正样本的，且IoU小于0.3的“hard negatives”，提高了定位的准确度。  

 3.hard negatives:    

　在训练过程中会出现 正样本的数量远远小于负样本，这样训练出来的分类器的效果总是有限的，会出现许多false positive。
　采取办法可以是，先将正样本与一部分的负样本投入模型进行训练，然后将训练出来的模型去预测剩下未加入训练过程的负样本，
　当负样本被预测为正样本时，则它就为false positive，就把它加入训练的负样本集，进行下一次训练，知道模型的预测精度不再提升
　这就好比错题集，做错了一道题，把它加入错题集进行学习，学会了这道题，成绩就能得到稍微提升，把自己的错题集都学过去，成绩就达到了相对最优
         

# 参考   
1、论文参考：        
   https://www.computer.org/csdl/proceedings/cvpr/2014/5118/00/5118a580-abs.html          
2、代码参考：     
   https://github.com/Liu-Yicheng/R-CNN
   https://github.com/yangxue0827/RCNN     
   https://github.com/edwardbi/DeepLearningModels/tree/master/RCNN          
3、博客参考：       
   http://blog.csdn.net/u011534057/article/details/51218218        
   http://blog.csdn.net/u011534057/article/details/51218250        
