# 英中机器文本翻译

评测英中文本机器翻译的能力。机器翻译语言方向为英文到中文。


## 依赖

- Python 3.5.2
- PyTorch 1.0

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

下面第一行是英文例句（数据集），第二行是人翻中文例句（数据集），第三行是机翻（本模型）中文句子（实时生成）。

<pre>

< because there  s still a chance that you might be alive tomorrow
= 因为你还有希望可能明天还活着，
> 因为还有可能你明天还活着，
< oh   man   do n t worry about that   man  . ai n t nobody gon na take you nowhere  .
= 哦，伙计，别担心那个。没人会带你走的。
> 哦，伙计，别担心，没人会带你去任何地方。
< we  ll ring you again when we can get to the next pit stop  .
= 我们到下一个休息站再打给你。
> 我们到下一个地方再打给你。
< i do n t know why we are listening to you at all  .
= 我就不知道我们为什么还要在这儿听你讲话了。
> 我不知道为什么我们要听你的。
< i may not have been here long   but i  ve learned one thing  .
= 我在这里的时间也许不长，但我至少明白了一样。
> 我可能来不了多久，但我学到了一件事。
< hey   hey   get off me  .   what are you doing  ?
= 嘿，嘿，别碰我。- 发生什么了？
> 嘿，嘿，别碰我。- 你在干什么？
< and always talking about art and books and movies and everything  .
= 他总是谈论艺术、书籍、电影和所有的东西。
> 总是谈论艺术和书。
< you  re doing this for the same reason i  m doing this  .
= 你做这些的原因是和我一样的。
> 你这么做是为了同样的原因。
< it  s a warm country  . oh   captain   would n t you like to come along too  ?
= 是个友好的国家。船长，你不来吗？
> 是个温暖的国家。哦，上尉，你不想一起去吗？
< i  m gon na need a little help  . but you said you could do it  .
= 我需要一点帮助。你说过你行的。
> 我需要帮助。但你说你能做到的。

</pre>