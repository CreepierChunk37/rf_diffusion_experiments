'model.py'中存放了RFM模型的实现；

'train.py'中存放了模型的训练代码，我对时间网格做了分批计算，防止OOM，调整't_batch_size'参数就行；

'sampling.py'中存放了采样代码，包括了线性时间调度和指数时间调度；

'other_methods.py'存放了不同parameterize的训练和采样代码；

把优化器修改成SGD就是梯度下降。
