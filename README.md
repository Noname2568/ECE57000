1. The dataset can be downloaded by this link:https://cloud.tsinghua.edu.cn/d/cdcdab829e184a698b63/  
  
2. The python file called ctrate_reid_database in database is used to split the dataset into training and testing set with some relating information.  
  
3. After the dataset is downloaded, it should be put into the database file and parallel to the ctrate_reid_database.
  
4. The main.py contains the specific setting of pre-train processing.
  
5. Util file define the loss function when the model processing the LReID.
  
6. The train.sh and run.sh are the bash shells used in Linux system which motivates me to finish this project in virtual machine to use linux system. However, as I metioned in term paper, the code use CUDA which need the GPU to accelerate the processing. In virtual machine, there is no real GPU so that the processing will be interrupted. To solve this problem, I tried to re-write the main.py to make the whole project can be run without train.sh which means the project can be run in Windows system. However, it seems failed. So the re-implementation is incomplete. The main(adjusted).py is what I tried to modified even though it failed.  
