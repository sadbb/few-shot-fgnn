# Fuzzy Graph Neural Network for Few-Shot Learning
Implementation of FGNN on Python3, Pytorch 1.1.0  

## Mini-Imagenet [\[Download Page\]](https://drive.google.com/drive/folders/1uZL6dhO-czXHYv_MR2HlrBU13q108Czr)    
Download dataset and copy it inside ```data``` directory:  

    
    .
    ├── ...
    └── data                    
       └── mini                
          ├── train  
          ├── val 
          └── test 
             ├── n01855672
             ├── ...
             └── n09256479

 
## Training  
```
# 5-Way 1-shot  
python3 main.py --exp_name $EXPNAME --dataset mini_imagenet 

# 5-Way 5-shot  
python3 main.py --exp_name $EXPNAME --dataset mini_imagenet --shot 5  
```

