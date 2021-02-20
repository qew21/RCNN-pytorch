import os

Image_size = 224
Staircase=True
cuda = True

Train_list = r'./train_list.txt'
Finetune_list = './fine_tune_list.txt'
DATA = './FlowerData'
Fineturn_save = './FlowerData/Fineturn'
SVM_and_Reg_save ='./FlowerData/SVM_and_Reg'
Out_put = './output'

T_class_num = 17
T_batch_size =256
T_learning_rate=0.0001
T_epoch=100
T_model=r'./output/alexnet.pth'

F_class_num = 3
F_batch_size = 256
F_learning_rate=0.0001
F_fineturn_threshold =0.3
F_svm_threshold =0.3
F_regression_threshold =0.6
F_epoch=50
F_model=r'./output/fineturn.pth'

R_class_num = 5
R_batch_size = 512
R_learning_rate=0.0001
R_epoch=5000
R_model=r'./output/regbox.pth'
