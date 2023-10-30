[📘使用文档]() |
[🛠安装教程]() |
[👀模型库]() |
[🆕更新日志]() |
[🚀进行中的项目]() |
[🤔报告问题]()

</div>

 ## 简介
mmgeneration_add是对mmgeneration的补充项目，和众多add系列项目一样，它旨在添加一些mm中未收录的额外的方法，但是又按照mm系列的框架来看，为了复用
包括runner在内的众多特性，此外mmgeneration的更新，只需要等价替换mmgen目录即可，configs按照现实工程进行配置，去掉mm多余的学术味。

<details open>
<summary>主要特性</summary>

- **便捷**     
    不需要改动原始mmgen的代码
    

</details>

## Tips
- **mmgeneration需要编译**   
pip install -r requirements.txt    
pip install -v -e .

- **关于离线下载的评测权重，可以离线去生成，也可以线上生成，GAN的测评多是外部权重加载**       
mmgeneration_add/mmgen/models/architectures/fid_inception.py中的inception模型自己离线下载一波     
mmgeneration_add/mmgen/core/evaluation/metrics.py中的TERO_INCEPTION_URL自己离线下载一下    
mmgeneration_add/mmgen/core/evaluation/metrics.py中的PR中的vgg16的权重851行，自行加载权重  
mmgeneration_add/mmgen/core/evaluation/metrics.py中的_load_inception_torch中的inception_v3的权重122行，自行加载权重   
mmgeneration_add/mmgen/models/architectures/lpips/pretrained_networks.py中16行vgg权重加载   
mmgeneration_add/mmgen/models/architectures/lpips/perceptual_loss.py中40行pnet_rand改为True    
mmgeneration_add/mmgen/models/architectures/lpips/perceptual_loss.py中52行自行加载权重     

- **G和D的交替训练**    
train_cfg=dict(disc_steps=8,gen_steps=1) 训练8次判别器，训练1次生成器      

- **图像转译**       
pix2pix和cyclegan这两个基本继承了原始mm中已有的几种场景，建筑图转换/斑马转换/季节转换/卫星图转换/线描谢图....     
[开源数据](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)        

- **configs的命名风格**   
{model}_[model setting]_{dataset}_[batch_per_gpu x gpu]_{schedule}    

- **线上训练报错**      
ModuleNotFoundError: No module named '_lzma'??     
sudo yum install xz-devel -y    
sudo yum install python-backports-lzma -y    
将requirements下的lzma.py cp到/usr/local/python3/lib/python3.6下，并安装pip install backports.lzma    

- **多卡训练**    
python -m torch.distributed.launch   --nproc_per_node=4   --nnodes=1 --node_rank=0     --master_addr=localhost   --master_port=22222 	train.py        
apis/train.py 中116行中更换mm的训练    

- **styleganv3**    
升级gcc 5.4，sudo pip install ninja,export CXX=g++     
[filtered_lrelu](https://blog.csdn.net/DavieChars/article/details/121857265 ),cpp17的标准，以在项目中更改     

- **layoutgan/const layout**      
/usr/local/cuda/lib64     
torch_geometric:图神经网络的框架，基于pytorch，https://zhuanlan.zhihu.com/p/142948273     
layoutgan并不是单纯的从噪声图中生成线框图    
layoutgan/const_layout https://gitee.com/leeguandong/const_layout,原作者代码比较完整    

- **styleclip**   
mmgeneration_add/mmgen/models/architectures/arcface/id_loss.py 第13行和23行中的arcface权重加载更换     

- **mmgen_add**  
mmgen_add对于layout系列的代码开发做不到原汁原味，只能说嵌入到同一个框架下，修修补补,粒度不会太细    
  
- **paddleGan cuda10.1**    
以下操作仅限在平台cuda10.1环境中，最终可以单卡，但不可以单机多卡训练
cd /usr/local/cuda-10.1/lib64       
ll -h        
sudo ln -s libcublas.so.10.0 libcublas.so.10.1    
sudo ln -s libcublas.so.10.1 libcublas.so     
缺少的libcublas.so可以用libcublas.so.10.1的代替         
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64/   (在启动的跑代码的terminal中设置)      

libnccl.so.2.4.8在/usr/local/lib中      
sudo ln -s libnccl.so.2.4.8 libnccl.so      
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64/         

在平台裸机cuda10.0环境中      
1.升级gcc5.4
2.安装cuda10.1，即将平台中cuda的软连接换成新的cuda10.1
unzip cuda-10-1.zip -d $HOME/local_disk/
sudo rm -rf /usr/local/cuda
sudo ln -s $HOME/local_disk/cuda-10.1  /usr/local/cuda
echo "export PATH=$PATH:/usr/local/cuda/bin" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/" >> ~/.bashrc
source ~/.bashrc
3.使用cuda9.0环境中/usr/local/cuda-9.0/lib64/libcublas.so.9.0.480中cp到libcublas.so.9.0.480 /home/ivms/local_disk/cuda-10.1/lib64/，sudo ln -s libcublas.so.9.0.480 libcublas.so
4.CUDA_VISIBLE_DEVICES=0,1 python -m paddle.distributed.launch main.py  

-- **paddleGan中权重更改**
paddlepaddle/PaddleGAN-develop/ppgan/metrics/fid.py中第31行INCEPTIONV3_WEIGHT_URL




## mmgen_add中添加方法
- ✅ [WGAN-div](https://arxiv.org/abs/1712.01026)
- ✅ [layoutgan](https://blog.csdn.net/u012193416/article/details/125716540?spm=1001.2014.3001.5501)
- ✅ [constlayout](https://blog.csdn.net/u012193416/article/details/125722049?spm=1001.2014.3001.5501)  




## 鹿班banner图合成


## 背景图合成
### 1.视频生成背景图生成



## 形状数据集合成


## 风格迁移


## 元素生成
mcb-1/2



