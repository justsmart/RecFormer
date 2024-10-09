# RecFormer
For paper "Information Recovery-Driven Deep Incomplete Multiview Clustering Network", accepted by TNNLS.

This code is implemented by python 3.9.13 and pytorch 1.13.0


"handwritten-5view" dataset is provided for a demo! 

Run "python train.py".

### If you are interested in our proposed new dataset [Aloi_deep](https://drive.google.com/drive/folders/1SIu_QJWJ0Jhqsb1IJR7sMQMcPGDiDYhz?usp=share_link), you can download it into 'data/'

You can use the function in construct_incomplete_index.py to construct the incomplete index matrix. And you can store the matrix in .mat file or other files, or you can construct the matrix in mydatasets.py (some simple modifications should be applied by youself) before training.


# Citation
```bibtex
@article{liu2023information,
  author={Liu, Chengliang and Wen, Jie and Wu, Zhihao and Luo, Xiaoling and Huang, Chao and Xu, Yong},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Information Recovery-Driven Deep Incomplete Multiview Clustering Network}, 
  year={2023},
  volume={},
  number={},
  pages={1-11},
}
```

If you have any questions, please contact me: liucl1996@163.com
