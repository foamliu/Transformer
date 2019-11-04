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

< but after our lone witness went missing   no charges were ever filed  .
= 但当我们的唯一证人失踪后，案件没有新的进展。
> 但在我们孤独的证人失踪后没有起诉。
< my son is on this list  . his name is right there  .
= 包括我儿子他也在死者名单上。
> 我儿子在名单上。他的名字就在这里。
< is there a decent spot around here to watch it  ?
= 有没有个合适的地方让我们看比赛？
> 有什么好地方可以看吗？
< and forgive us our sins   as we forgive those that sin against us  .
= 你宽恕了我们的罪孽，如同我们原谅那些得罪我们的人一样。
> 原谅我们的罪恶，因为我们原谅那些对我们不利的罪。
< aii the computers are down  . we even had some trouble finding your records  .
= 所有电脑都不能工作了。我们差点找不到你的档案。
> 所有电脑都坏了。我们甚至找不到你的记录。
< i  m telling you   you  re gon na order the goddamn chicken sandwich  .
= 我告诉你，你要点他妈的鸡肉三明治。
> 我告诉你，你要点鸡肉三明治。
< oh for the love of pete  ! go on to the bushes and do your business  .
= 噢看在老天的份上！到灌木丛里解决掉吧。
> 噢，看在彼得的份上！去树林吧。
< do n t worry  . i  ll get you two where you need to go  . you  ve earned it  .
= 别担心。我会带你们到要去的地方。这是你们应得的。
> 别担心。我会带你们俩去你应得的。
< i know she has children   so i assumed she had some level of independence  .
= 我知道她有小孩，所以我想她有一定的独立能力。
> 我知道她有孩子，所以我想她有自己的独立。
< i almost forgot  . she told me to chew you out  .
= 我差点忘了。她要我教训你。
> 我差点忘了。她让我咬了你。

</pre>