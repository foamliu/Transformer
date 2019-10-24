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

< i mean   i ca n t believe it  s really me up there  .
= 我是说，我不敢相信那真的是我。
> 我是说，真不敢相信我真的在那里。
< they do n t know who they are or where they  re going  .
= 他们不知道他们是谁不知道要去哪里。
> 他们不知道他们是谁或者去哪。
< you mean you want me to keep a secret  ? yes  .
= 你要我帮你保密？对。
> 你想让我保守秘密？是的。
< where do you want my hands  ? in front of me   or behind me  ?
= 想把我的手绑在哪？前面还是后面？
> 你要我的手呢？在我面前，还是我后面？
< i  ll be ready to take on scene command in three   two   one  . i  m in charge  .
= 我准备接手现场指挥倒数三，二，一。现在由我负责。
> 我准备好进入3号，2，1。我负责。
< but i  m afraid the darkest hours of hell lie before you  .
= 但恐怕地狱中最黑暗的时刻正在等待着你。
> 但我担心在你面前最黑暗的时间。
< and you have to comfort yourself with the fact that
= 你必须安慰自己面对现实，
> 你得安慰自己，
< deep down   i know you do n t really like to try
= 从下面进去你就是不愿意去试一下-
> 我知道你不喜欢尝试，
< sa   yong   nobody will blame you if we lose  .
= 如果我们输了，没人会怪你的。
> 如果我们输了，没人会怪你。
< it  s going well   thanks to you  . i  ll talk to you later  .
= 托您的福，都很好。我一会去找您。
> 一切都很好，谢谢你。我晚点再跟你谈。


</pre>