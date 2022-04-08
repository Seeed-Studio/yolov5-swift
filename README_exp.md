- 模型配置文件位置：./configs/或./configs/_base_/models
- 数据配置文件位置：./configs/或./configs/_base_/datasets
- 学习率配置文件位置：./configs/_base_/schedules

-  模型代码位置：./mmdet/models
-  数据预处理代码位置：./mmdet/datasets
-  通用工具代码位置：./mmdet/core

## 一、配置网络结构
-  前提条件：配置网络结构前需编写模型各个组件可以进行配置的代码。
### 1.网络配置
```python
model={
    type=模型类名
    backbone={
        type=backbone类名
        backbone类各种参数
    }
    neck={
        type=neck类名
        neck类的各种参数
    }
    bbox_head={
        type=head(检测头类名)
        head类的各种参数
    }
    train_cfg={
        训练时检测头的配置
    }
    test_cfg={
        测试或验证时检测头的配置
    }
}	

```


## 二、训练参数配置
### 1.位置：./configs/
### 2.数据增广配置
```python
train_pipeline=[
    dict(
        type=需要执行的操作
        训练时数据增广方面的各种参数
        )
    ......
    ]
test_pipeline=[
    dict(
        type=需要执行的操作
        测试时数据增广方面的各种参数
    )
......
    ]
```


### 3.数据集加载配置
```pytohn
train_dataset={
    type=数据集加载类名
    dataset={
        type=数据集加载根类
        ann_file=数据集标签文件路径
        img_prefix=数据集图像文件路径
    }
}
```
### 4.数据加载器配置
```python
data={
    batch-size=每次加载数据量
    workers=数据加载进程数
    persistent_workers=数据加载完后是否关闭进程
    train=训练数据增广配置
    val=验证数据增广配置
    test=测试数据增广配置
}
```
### 5.优化器配置
```python
optimizer={
    type=优化器类名
    优化器参数
}
```

### 6.学习率配置
```python
lr_config={
    Policy=学习率策略
    warmup=预热阶段学习率调整策略
    warmup_iter=预热阶段迭代次数
    warmup_ratio=预热阶段学习率
}
```
## 三、模型训练
### 1.训练参数
-  config：配置文件路径
-  work-dir：模型权重和训练日志保存位置
- resume-from：需要继续训练的模型权重文件路径
- auto-resume：加载最后一次保存的权重文件
- gpu-id：需要使用gpu设备号
- seed：设置全局随机种子
- deterministic：是否使用卷积优化，配合随机种子使用便于复现
- cfg-options：覆盖或合并配置文件内的参数设置，格式xxx=yyy
- launcher：是否使用分布式训练，如果用用何种框架
- local_rank：本机节点的优先级

### 2.初始化流程
1) 读取配置文件		>>>  config
2) 进程数设置			>>> \_base_:0
3) 建立工作文件 		>>> work-dir
4) 设置随机种子 		>>> seed 和 deterministic
5) 构建模型			>>> config.model
6) 初始化模型权重		>>> model.init_weights()
7) 构建数据集			>>> config.data.train ==  config.train_dataset
8) 构建数据加载器		>>> dataset config.data,gup_id,seed,
9) 构建优化器			>>> model,config.optimizer
10) 构建训练器		>>> config.runner,model,optimizer,config.work_dir,log
11) 注册hook			>>> config.lr_config,
12) 开始训练			>>> datal_oaders,config.workflow