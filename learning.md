1.model.train和model.eval   https://blog.csdn.net/qq_38410428/article/details/101102075

2.多GPU分布式训练  https://blog.csdn.net/qq_34914551/article/details/103236807

3.BN的问题 

https://zhuanlan.zhihu.com/p/65439075

evaluation的时候要重新统计BN，bn的running statistics 需要重新统计一遍

BN会保存训练的均值和方差并求滑动平均来作为eval时候的均值和方差，这个本来是准的，但是在supernet里，统计的是多条路径的均值和方差，所以对于单条路径是不准的，需要重新统计单条路径的
