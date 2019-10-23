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

< oh   all right  . well   i  ll be   i  ll be this guy  .
= 哦，好吧。嗯，那我，那我扮他吧。
> 哦，好吧。好吧，我会的。
< you do n t want to go upstairs   cause you get scared  .
= 你害怕所以不想去。
> 你不想上楼，因为你害怕了。
< you do n t get to say what she was  . neither do you  !
= 你没权利评论她是怎么一个人。你也没！
> 你不能说她是谁。你也是！
< if we do n t disable them   they  re just going to fly away
= 如果我们不干掉它们，那它们就会飞走，
> 如果我们不阻止他们，他们就会飞，
< do n t take it personally   man   but i got ta make a living  .
= 别以为这是你一个人的事业，哥们儿，我也要靠这过活的。
> 别这么说，伙计，但我得活下去。
< i seem to have lost an integral part of my plan  .
= 我好像丢失了计划里的重要部分。
> 我好像失去了我计划的一部分。
< what would you do if you were n t out making yourself a better citizen  ?
= 如果你不是外面，你会做什么成功的因素你自己一个较好的市民？
> 如果你不做更好的公民你会怎么做？
< i know  . they were at the lab a half   hour ago  .
= 我知道。他们半小时前在冲洗室。
> 我知道。他们半小时前在实验室。
< the day after her kid kicked the crap out of my limo  .
= 她女儿对我车窗又敲又砸的后一天。
> 当她的孩子踢了我的车的那天。
< you want it to go better than today   you need to get some rest  .
= 你要想明天过得比今天好，就需要好好休息一下。
> 你希望今天能更好一点，你需要休息一下。


</pre>