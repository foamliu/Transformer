# 英中机器文本翻译

评测英中文本机器翻译的能力。机器翻译语言方向为英文到中文。


## 依赖

- Python 3.6.8
- PyTorch 1.3.0

## 数据集

我们使用AI Challenger 2017中的英中机器文本翻译数据集，超过1000万的英中对照的句子对作为数据集合。其中，训练集合占据绝大部分，为12904955对，验证集合8000对，测试集A 8000条，测试集B 8000条。

可以从这里下载：[英中翻译数据集](https://challenger.ai/datasets/translation)

![image](https://github.com/foamliu/Transformer/raw/master/images/dataset.png)

## 用法

### 数据预处理
提取训练和验证样本：
```bash
$ python pre_process.py
```

### 训练
```bash
$ python train.py
```

要想可视化训练过程，在终端中运行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### Demo
下载 [预训练模型](https://github.com/foamliu/Scene-Classification/releases/download/v1.0/model.85-0.7657.hdf5) 然后执行:

```bash
$ python demo.py
```

下面第一行是英文例句（数据集），第二行是人翻中文例句（数据集），之后一行是机翻（本模型）中文句子（实时生成）。

<pre>

< there are places in the world where girls do n t get educated   simply because they are girls  .
= 在这个世界上某些地方的女孩，不能得到教育仅仅因为她是女孩。
> 世界上有一些女孩没有受过教育，仅仅是因为她们是女孩。
< i  ve noticed this black van parked outside the house every day  .
= 我注意到有辆黑色的车每天都停在房子外面。
> 我每天都注意到一辆黑色货车停在房子外面。
< i mean   i feel like i  m okay  cause i passed on the crazy  .
= 我是说，我觉得不错因为已经从疯狂里解脱了。
> 我是说，我感觉很好因为我错过了疯狂。
< i do n t know  . i was having this dream  . it  s all right  . i  m here now  .
= 我不知道。我刚才做了个梦。没事了。我在这里。
> 我不知道。我做了个梦。没关系。我在这儿。
< and if the girls see you up here   they  re gon na take it
= 如果她们看到你坐这她们会觉得
> 如果女孩们看到你在这里，他们会接受的
< getting filled out like an application constitutes being with a guy  .
= 愿意跟小男孩在一起。
> 就像和男人在一起一样。
< every single brother in my fraternity has worn this suit  .
= 兄弟会里每个人都穿成这样。
> 我兄弟会的每个兄弟都穿这件衣服。
< but i  m here in town   checking out some real estate   and
= 但我进城来看房子，以及-
> 但我在城里，查一些房地产，
< hey   i was gon na come and see you today and say hi  .
= 嘿，我今天打算过去看看你打个招呼。
> 嘿，我今天想去看你然后打个招呼。
< it was a good show  . stop saying it was a good show  . shh  !
= 今晚的节目很好。不要再说它好了！
> 是个好节目。别说是好节目了。嘘！


</pre>