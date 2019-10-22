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

下面第一行是英文例句（数据集），第二行是人翻中文例句（数据集），之后五行是机翻（本模型）中文句子（实时生成）。

<pre>

< you say that shit   or i  ll walk out this fucking door  .
= 你再不说，妈的那我就下车。
> 你这么说，不然我就走。
< yes   a little   sir  . tell her to visit  . with a friend  .
= 对，有点想，长官。让她来看你啊。就说是朋友。
> 是的，有一点，先生。让她去见一个朋友。
< boy   how am i gon na tell my mom about this  ?
= 天啊，这叫我要怎么跟我妈妈说啊？
> 孩子，我该怎么告诉我妈妈？
< we need to know it  s there  . we need to know it  s safe  .
= 我们只要知道它在那儿。我们只要知道它很安全。
> 我们要知道它在那儿。我们要知道它安全。
< before this race and meeting the public and all that
= 在比赛，与公众见面和这些之前，
> 在比赛结束之前，
< and i  ve never been on one with a woman before  .
= 我也没坐过有女人的船。
> 我从没和一个女人在一起。
< okay   i do n t want you in here anymore  . please leave  .
= 我不想你在这儿了。请离开。
> 好吧，我不想你再进来。请离开。
< did you say something about us to her or not  ?
= 你到底告不告诉我们她是谁？
> 你跟她说了什么吗？
< do n t be  . you did n t do it  .   you did n t   either  .
= 别这样。你没错，你也没有。
> 别这样。你没做。- 你也没有。
< were you aware that she was twice treated for alcohol addiction  ?
= 你知道她曾两次为酗酒问题做过治疗吗？
> 你知道她有两次吃了两次？

</pre>