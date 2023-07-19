# CrackSeU
This repository is the official implementation of the **Crack** **Se**gmentation **U**-shape (**CrackSeU**) Network.

### 🔥 Break News:  
Our paper is finally accepted by Automation in Construction after a year of review. I have to say it has been a long and tough journey. 😭.

The paper is available:  
[Online monitoring of crack dynamic development using attention-based deep networks](https://www.sciencedirect.com/science/article/pii/S0926580523002820), Automation in Construction, 154 (2023) 105022, by Wang chen*, [Zhili He](http://zl-he.com/)*, and Jian Zhang#. ( *: Co-first authors, #: Corresponding Author )

![Framework](figures/Framework.png)

# Getting Started
### 1. Requirement
~~~
Recommended versions are
    * python = 3.5
    * pytorch = 1.12.1
    * CUDA 11.6.2 and CUDNN 8.6.0  
Other requirements can be found in the Requirements.txt.
~~~

### 2. Installation
```bash
git clone https://github.com/hzlbbfrog/CrackSeU
cd CrackSeU
pip install -r Requirements.txt
```
Or, you can directly "Download ZIP".

### 3. Build your own dataset
You can refer to the following file tree to organize your own data.
```
Your project
│   README.md
│   ...
│   CrackSeU_main.py
│
└───Dataset
    |
    └───Your dataset name
        |
        └───Train
            └───images
            └───masks
        └───Test
            └───images
            └───masks
│  
└───...Other directories   
```

### 4. Training
To train the CrackSeU with LN_VT, simply run:
```shell
python CrackSeU_main.py --action=train --arch=CrackSeU_S_LN_VT --epoch=50 --batch_size=2 --lr=1e-4
```

### 5. Test
To test the CrackSeU with LN_VT, simply run:
```shell
python CrackSeU_main.py --action=test --arch=CrackSeU_S_LN_VT --test_epoch=50
```

# Method
### :rocket: The network architecture of CrackSeU:
![CrackSeU](figures/CrackSeU.png)

### :rocket: Illustration of the proposed FFM:
![FFM](figures/FFM.png)

# Citing CrackSeU
You are very welcome to cite our paper! The BibTeX entry is as follows:

```BibTeX
@article{CrackSeU,
title = {Online monitoring of crack dynamic development using attention-based deep networks},
journal = {Automation in Construction},
volume = {154},
pages = {105022},
year = {2023},
doi = {https://doi.org/10.1016/j.autcon.2023.105022},
url = {https://www.sciencedirect.com/science/article/pii/S0926580523002820},
author = {Wang Chen and Zhili He and Jian Zhang},
keywords = {Crack identification, Online monitoring method, Deep learning}
}
```

# Acknowledgements
SEU is also the abbreviation of [Southesast Univertisy](https://www.seu.edu.cn/).  
The name of our framework ( Crack[**SeU**](https://www.seu.edu.cn/)) is also dedicated to the 120th anniversary of Southeast University.

