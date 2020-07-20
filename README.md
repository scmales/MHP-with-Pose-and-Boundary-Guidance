# MHP-with-Pose-Boundary-Guidance
### Network
![Network](https://github.com/scmales/MHP-with-Pose-and-Boundary-Guidance/tree/master/images/network.png)
### Requirement
Python 3.6<br>
Pytorch 0.4.1 + related inplace_abn<br>
...<br>
(requirement.txt)
### Dataset
├── pascal_person_pose_and_part <br>
│ ├── JPEGImages <br>
│ ├── pascal_person_part_gt<br>
│ ├── pose_annotations<br>
│ ├── train_id.txt<br>
│ ├── val_id.txt<br>
### Training and Evaluation
```
./run_train.sh
```
```
./run_evaluate.sh
```
 
