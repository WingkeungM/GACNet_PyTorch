# GACNet_PyTorch
This is the pytorch implmentation of GACNet on S3DIS.
<p align="middle">
  <img src="flowchart.jpg">
</p>

If you find this repository useful. Please consider giving a star :star:.
## Dependencies
- Python 3.6
- PyTorch 1.7
- cuda 11


## Install pointnet2-ops

  ```
  cd pointnet2_ops_lib
  python setup.py install
  ```
  
## Train Model
  ```
  cd tool
  python train.py
  ```

## Test Model
  ```
  cd tool
  python test.py
  ```
  
## References
This repo is built based on the Tensorflow implementation of [GACNet](https://github.com/wleigithub/GACNet). Thanks for their great work!
