1.scheduler
根据总的epoch数来调整lr的变化，每个step会降学习率，正常的训练，最后的学习率应该是最开始的百分之一或者千分之一，最开始也可以有个短的warmup，从小学习率升到最高

2.训练中动态调整学习率lr，optimizer.param_groups
https://blog.csdn.net/bc521bc/article/details/85864555

3.git rebase
https://www.yiibai.com/git/git_rebase.html
