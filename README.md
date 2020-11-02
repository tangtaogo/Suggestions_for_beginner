# Suggestions_for_beginner

### learning2020

1.Batchsize,lr,gpu_nums 一定得对应上

2.optimizer_step，可以增大batchsize，但是不影响显存

loss = criterion(logits, target)
loss = loss / frequence
loss.backward()
if batch_iter % frequence == 0 or last_batch_iter:
  optimizer.step()
  optimizer.zero_grad()
  
 frequence = 7，这个是每7步才step一次optimizer

3.看效果可以用tenserboard检查一下存下来的feature map https://www.cnblogs.com/tengge/p/6376073.html

tensorboard --logdir=logs

4.git教程https://blog.csdn.net/HcJsJqJSSM/article/details/84558229

私有库要 git clone http://用户名:密码@github.com/changlin31/SelfSupDNA.git

git clone  git@github.com:Trent-tangtao/LearningGit.git

git add hello.txt

git commit -m "添加了hello.txt"  +   git push origin master

git stash

git branch -a(查看所有分支包括本地分支和远程分支).

git checkout -b branchname 本地创建新的分支.(直接新建一个分支然后切换至新创建的分支).就是创建加切换分支. 等价于命令：git branch branchname+git checkout branchname.

git push origin branchname 将新分支推送至GitHub 

git checkout master

git pull

git checkout tangtao

git rebase

5.跨卡同步BN

6.pytorch hook

7.@ Python装饰器

8.Linux ln（英文全拼：link files）命令是一个非常重要命令，它的功能是为某一个文件在另外一个位置建立一个同步的链接。


9.find_unused_parameters=True

### 关于怎么研究一个方向的：

0.可以先直接搜索一下相关博客等了解一下这个方向是干什么的，前景和意义，有个大概的定位和了解

1.1搜寻相关的综述，可以先看简单的博客综述总结，或者知乎专栏，主要了解方向的：起源，传统方法，深度学习方法，以及现在常用的方法和进展，了解现在能做的最好方法思路的performance以及瓶颈在哪里(了解坑的方向以及坑的深浅)，通常各种会议或者论坛也会有相关报告，可以直接B站搜相关方向的视频，很建议B站上多看看课程和报告

1.2在前面基础的了解下，可以看论文的survey，通常survey里面方法比较详细，比博客要好，能够更好的了解方向的整个发展

2.去找方向里面的milestones，通常survey里面也会列出时间线和相关重要的里程碑的论文，可以细看里程碑式论文的方法，通常是一种新方法或者新思路新框架的出现，通常会引起一波潮流，是后面一阵论文的基础

3.了解方向最新的动态，最新的顶会CVPR，ECCV等上面的相关论文，是不是基于前面里程碑论文的改进，还是新的思路，同时关注最新的论文的performance和瓶颈

TIPS：

1.读论文或者看综述的时候，注意一下团队和作者，一方面对大佬心生膜拜，另一方面擅长相关方向的团队，可能后续也有有一系列前沿的工作，可以持续关注，此外以后也可能有机会过去学习也不一定；

2.组会可能会有很多其他方向的，比如CV，可能有做细粒度的，有检测分割，可能开组会的汇报不是你的研究方向，虽然可能不怎么听得懂，但是多听绝对是会有好处的，经常会有一些想法或者方法可以借鉴，哪怕是attention也是从跨域的NLP借鉴过来的

### 代码和实验方面

1.github代码下载不下来，可以去国内的码云，然后导入github的项目，然后码云下载就很快；

2.跑代码一定要有checkpoints及时的存模型，最好也得有try&catch，方便及时打断运行时保存模型，log也要写好，尤其注意(时间，epoch，acc等)

3.传数据集，或者跑代码，一定要先试试小的mini的数据集或者先跑简单数据集的代码，防止跑了一两天发现跑错了，一定先试试

4.路径的问题：下面也提到服务器可能会后加磁盘挂载，所以可能数据的位置不容易轻易找到，所以你可以cd到相应的数据集文件夹，然后pwd出路径

5.shell跑实验时间过长容易断开，可以设置shell不断开，最好设置一下，然后此外我推荐的解决方案是用nohup命令或者tmux命令后台挂起，然后就可以去干自己的事情，例nohup python train.py &，这样输出都会存到nohup.out里面

新建 tmux new -s dna  

关闭：第一步：输入组合键 Ctrl+B，然后松开。第二步：输入字母 d。

重连 tmux ls  ;  tmux a -t dna

再创建一个窗口：第一步：按 Ctrl+B 组合键，然后松开。第二步：再单独按一下 c 键

假如我们要切换到 0：bash 这个窗口，步骤如下：第一步：按 Ctrl-B 组合键，然后松开。第二步：按数字 0 键。

删除：Ctrl+d 组合键

6.多卡的机器，通常tf和torch之间有些玄学不能共存的问题，最好都用torch；然后八块卡最好也别独占，可以设定自己用的GPU的块号 CUDA_VISIBLE_DEVICES=1 python train.py  或者代码中 import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

7.kill -9 来强制终止退出 

8.在Ubuntu下 使用Tab键报错：cannot create temp file for here-document: no space left on device
解决办法：rm -rf /tmp/*

9.find ./ -name ‘xxx.*’ -exec rm {} ;

10.mv /home/tangtao/DNA /media/Disk/tangtao/


### 服务器方面

1.ssh  用户名@服务器IP ，例如ssh tangtao@210.20.96.136 

然后输入密码就可以

2.服务器IP应该是只有校园网才能连，远程得先有VPN

3.本地与服务器之间文件和代码的上传和下载

文件的话，scp命令即可，要是习惯桌面，windows上推荐软件winscp，mac上推荐Cyberduck，传输协议SFTP；代码的话，可以当做文件处理，本地改好再传，要是觉得不方便，可以利用pycharm专业版连接服务器，直接联通本地和服务器；

PS：scp例子，rsync可以断点续传，也可以挂后台

文件夹  scp -r   /Users/tang/Desktop/test   tangtao@172.18.167.4:data/imageNet2012/

文件  scp -r   /Users/tang/Desktop/temp.txt   tangtao@172.18.167.4:temp.txt

rsync -P --rsh=ssh /Users/tang/Desktop/ILSVRC2012_img_val.tar tangtao@172.18.167.4:data/imageNet2012/ILSVRC2012_img_val.tar

4.shell

mac和ubuntu就原生的shell的就非常棒了；windows上其实winscp也有，个人推荐windows上用ubuntu for windows，在windows应用商店下载就行，或者其他的软件类似xshell

5.服务器满了

正常给你开的服务器是在home下的，home分的盘可能会因为用户的增加而不够用，这时候你可能传数据也传不上，存模型也存不了；所以通常服务器都会后加磁盘，你可以cd ..到home，然后cd .. 到主目录，主目录下可能就会有data(原始)，data1，data2等等磁盘，进去mkdir一个文件夹使用就可以啦

PS:计算一个文件夹，比如data1大小 du -h  以及 剩余空间 df -hl

6.watch  nvidia-smi


### 组会、报告、周报等方面

PPT或者报告，如果是引用的别人的工作，PPT当页，或者报告的最后，一定给出链接和作者；

### 其他

1.和师兄师姐老师们搞好关系，多动手实验，不懂就多问，不用不好意思

2.做好备份

### 其他命令

##### conda&list

conda安装：建议清华源下载.sh文件，然后sh  xxxx.sh，注意中间有几个让你选择的过程，看完手册，安装目录yes就行，安装完有个东西一点注意，就是是否添加到环境变量，默认是NO，别按回车，输入yes，这样后面就不用自己添加到环境变量了，最后一个让你是否安装vscode，选no就可以；安装好之后，重启一下shell就行，然后输python，应该就是conda的python了

conda env list

conda list

pip list

conda create --name xxxx python=3.x

source activate xxxxx(不加环境名，就是base)

source deactivate

conda的环境可以到导出yml文件:

conda env export > /home/tangtao/torch.yaml   conda env export > torch.yaml

conda env create -f torch.yaml

#### CV常用数据集

ImageNet2012 处理 val&train，见此项目

coco   json

voc  vos数据集自己制作的话，推荐使用labelImg





