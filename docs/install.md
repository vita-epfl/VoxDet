# Step-by-step installation instructions

VoxDet is developed based on the official CGFormer codebase and the installation follows similar steps.

**a. Create a conda virtual environment and activate**

python 3.8 may not be supported.

```shell
conda create -n voxdet python=3.7 -y
conda activate voxdet
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**

```shell
# Using pip is recommended

pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

or 

```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

We selected this PyTorch version because **mmdet3d 0.17.1** does not support **PyTorch ≥ 1.11**.

> **Note:** The following `pip` installation requires **CUDA 11.X** on your local machine.

- To avoid compilation issues, ensure your CUDA toolkit version matches the CUDA version used to compile PyTorch (e.g., 11.x).  
- Use `nvcc -V` to verify your CUDA version.  
- After compilation, you can use both CUDA 11.X and 12.X for training.

**c. Install mmcv, mmdet, and mmseg** 


```shell
pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
```

**c. Install mmdet3d 0.17.1 and DFA3D**

Compared with the offical version, the mmdetection3d provided by [OccFormer](https://github.com/zhangyp15/OccFormer) further includes operations like bev-pooling, voxel pooling. After this step, make sure mmdet3d appearing on your pip list.

```shell
cd packages
bash setup.sh
cd ../
```

**d. Install other dependencies, like timm, einops, torchmetrics, spconv, pytorch-lightning, etc.**

```shell
pip install -r docs/requirements.txt
```

**e. Fix bugs (known now)**

```shell
pip install yapf==0.40.0
pip3 install natten==0.14.6+torch1101cu113 -f https://shi-labs.com/natten/wheels
```
