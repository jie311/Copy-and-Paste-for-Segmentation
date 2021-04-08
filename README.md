# Copy-and-Paste
Simple Copy and Paste Implementation for Semantic Segmentation
</br>
</br>
Link to the [paper](https://arxiv.org/abs/2012.07177) </br>
</br>
</br>

### Current Status
- Tested. Works with albumentations. (See demo [here](https://github.com/WeiChihChern/copy-and-paste/blob/main/Example/Demo.ipynb "here"))
- Current implementation contains copy then paste only. Since semantic segmentation annotation may not be labeled as instance segmentation (instance wise annotated).

### Augmentation Flowchart:
1.  Put `SemanticCopyandPaste()` before other albumentations augmentation (See demo [here](https://github.com/WeiChihChern/copy-and-paste/blob/main/Example/Demo.ipynb "here"))
2. Then follow other augmentation such as flip, transpose, random crop, etc.


### Before and After
Before Augmentation:
![image](https://user-images.githubusercontent.com/40074617/113963987-9a385a00-97f8-11eb-8ee3-6c3f0bbdb426.png) </br>
After Augmentation:
![image](https://user-images.githubusercontent.com/40074617/113964055-be943680-97f8-11eb-803f-0756bfc99dbe.png)
