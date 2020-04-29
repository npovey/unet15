

```bash
[npovey@ka data]$ python3 -m venv torch_env
[npovey@ka data]$ source torch_env/bin/activate
(torch_env) [npovey@ka data]$ pip3 install tensorflow-gpu
(torch_env) [npovey@ka data]$ pip3 install torch
(torch_env) [npovey@ka pytorch_unet]$ pip3 install Pillow
```

```bash
(torch_env) [npovey@ka pytorch_unet]$ pip freeze > requirements.txt
```

requirements.txt 

```txt
absl-py==0.9.0
astor==0.8.1
gast==0.3.3
google-pasta==0.2.0
grpcio==1.28.1
h5py==2.10.0
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
Markdown==3.2.1
numpy==1.18.2
Pillow==7.1.1
protobuf==3.11.3
six==1.14.0
tensorboard==1.14.0
tensorflow-estimator==1.14.0
tensorflow-gpu==1.14.0
termcolor==1.1.0
torch==1.4.0
Werkzeug==1.0.1
wrapt==1.12.1
```

```bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=0
```

```bash
[npovey@ka ~]$ cd data/
[npovey@ka data]$ source torch_env/bin/activate
(torch_env) [npovey@ka data]$ cd pytorch_unet/
(torch_env) [npovey@ka pytorch_unet]$ 
```



```bash
(torch_env)[npovey@ka pytorch_unet]$ screen -S pytorch_unet
(torch_env) [npovey@ka pytorch_unet]$ python3 pytorch_unet.py > outputput.txt
```

