### 3-learning2020
1.git rebase：  rebase的话还是两个分支，只是把那些commit复制过来，cherry pick也是，现在rebase有很多冲突了，不合也行

https://www.yiibai.com/git/git_rebase.html

2.seed，随机数种子固定：seed不固定，会不会不同卡走的是不同路径？

### 2-learning2020

1.model.train和model.eval   https://blog.csdn.net/qq_38410428/article/details/101102075

2.多GPU分布式训练  https://blog.csdn.net/qq_34914551/article/details/103236807

3.BN的问题   https://zhuanlan.zhihu.com/p/65439075

evaluation的时候要重新统计BN，bn的running statistics 需要重新统计一遍

BN会保存训练的均值和方差并求滑动平均来作为eval时候的均值和方差，这个本来是准的，但是在supernet里，统计的是多条路径的均值和方差，所以对于单条路径是不准的，需要重新统计单条路径的

4.tensor.clone() 和 tensor.detach()  https://zhuanlan.zhihu.com/p/148061684

5.GPU和显存分析   https://zhuanlan.zhihu.com/p/31558973 https://blog.csdn.net/swocky/article/details/105922049 https://zhuanlan.zhihu.com/p/91485607

6.requires_grad和autograd.no_grad，固定部分参数进行网络训练
https://blog.csdn.net/g11d111/article/details/80840310  https://www.jianshu.com/p/fcafcfb3d887

7.查看预训练模型的参数 https://blog.csdn.net/feizai1208917009/article/details/103598233
pretrained_dict = torch.load(path)
for k, v in pretrained_dict.items():  # k 参数名 v 对应参数值
        print(k)

8.python中实现问号表达式 max = a if a > b else b  https://www.cnblogs.com/xxiong1031/articles/7099901.html

9.RuntimeError: Expected object of device type cuda but got device type cpu for argument #2 ‘target’ in call to _thnn_binary_cross_entropy_forward
https://blog.csdn.net/weixin_37913042/article/details/103009733

10.python的星号（*）和双星号（**）  https://www.cnblogs.com/empty16/p/6229538.html


### 1-learning2020

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

5.跨卡同步BN   Sync BN

6.代码里面：pytorch hook：帮你做一些代码主体之外的事情；

Registry：register就是相当于你在config里写函数名的字符串可以自动调用这个函数 

7.@ Python装饰器

8.Linux ln（英文全拼：link files）命令是一个非常重要命令，它的功能是为某一个文件在另外一个位置建立一个同步的链接。

 ln -s /data3/wubowen/imagenet data

9.find_unused_parameters=True

10.load那里，要加个strict＝False,因为改了model,要传到pytorch  nn.module的方法里，这个函数里面调用了self.load_state_dict之类的吧

https://blog.csdn.net/hungryof/article/details/81364487

11.resume和load不太一样，resume还会load optimizer，就是动量啥的都继续，正常应该都用resume

12.htop，看cpu占用率,https://linux.cn/article-3141-1.html

13.batchsize,不一定非得是32，64这些，可以随意

14.__init__.py 文件的作用是将文件夹变为一个Python模块,Python 中的每个模块的包中，都有__init__.py 文件

15.lamada匿名表达式，例如：lambda x,y:x+y

16.x.view(x.size(0), -1)   https://blog.csdn.net/whut_ldz/article/details/78882532

17.apex混合精度加速，https://blog.csdn.net/agq358/article/details/108404642
Upgrade gcc to version 7
$ sudo yum install centos-release-scl
$ sudo yum install devtoolset-7-gcc*
$ scl enable devtoolset-7 bash
$ which gcc
$ gcc --version
Install Apex
$ git clone https://github.com/NVIDIA/apex.git
$ cd apex
$ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

18.batchsize, step(iteration), epoch 含义, https://blog.csdn.net/wcy23580/article/details/90082221?utm_medium=distribute.pc_relevant.none-task-blog-title-2&spm=1001.2101.3001.4242

19.lr_scheduler：调整学习率,https://blog.csdn.net/qyhaill/article/details/103043637
