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

< i can give you time to discuss it   but this is a decision that needs to be made soon  .
= 我可以给你们时间讨论，但这个决定需要快点做。
> 我可以给你时间讨论，但这是很快决定。
> 我可以给你时间讨论，但这是个很快决定。
> 我可以给你时间讨论，但这是很快要决定。
> 我可以给你时间讨论，但这是很快要做的决定。
> 我可以给你时间讨论，但这是必须尽快做出决定。
< because we were lied to for a really long time
= 因为我们被你和一个同事
> 因为我们被骗了很久
> 因为我们被骗了很长时间
> 因为我们骗了很久
> 因为我们骗了很长时间
> 因为我们被骗了很久了
< i wanted all of us to really think about it  .
= 我要我们大家都考虑清楚。
> 我想我们大家好好想想。
> 我想我们都好好想想。
> 我想大家好好想想。
> 我想要大家好好想想。
> 我想要我们大家好好想想。
< so you see   i take a great deal of pride in what i do  .
= 我对我所做的事非常自豪。
> 你看，我为自己的所作所为感到自豪。
> 所以你看，我为自己的所作所为感到自豪。
> 你看，我为自己的所作所为感到骄傲。
> 所以你看，我为自己的所作所为感到骄傲。
> 你看，我为我所做的感到自豪。
< when he was a kid   he slept on the floor  .
= 他小时候，一直睡在地板上。
> 他小时候睡在地上。
> 他小时候，睡在地上。
> 他小时候，睡在地板上。
> 他小的时候，就睡在地上。
> 他小的时候，睡在地上。
< we were watching the news   we got your message   you  re all right  ?
= 我们刚刚看了新闻我们听到你留言，你真的没事？
> 我们在看新闻，我们收到你的留言，你没事吧？
> 我们在看新闻，收到你的留言，你没事吧？
> 我们在看新闻，我们收到你的信息，你没事吧？
> 我们在看新闻，我们收到你的留言了，你没事吧？
> 我们在看新闻，收到你的信息，你没事吧？
< i have n t seen your films  . you  re not the only one  !
= 我没看过你的电影。很多人都没看过！
> 我没看过你的电影。你不是唯一的一个！
> 我没看过你的电影。你不是唯一的！
> 我没看过你的电影。你不是唯一一个！
> 我没见过你的电影。你不是唯一的一个！
> 我没看你的电影。你不是唯一的一个！
< what  s happening here  ?
= 现在是什么情况？
> 怎么了？
> 出什么事了？
> 怎么回事？
> 这里发生了什么？
> 出了什么事？
< to truly love someone before i take her from you  .
= 我再把她夺走是什么样的感觉。
> 在我把她从你身边夺走真正的爱人。
> 在我夺走她之前我真的爱上一个人。
> 在我把她从你身边带走一个人之前。
> 在我把她从你身边带走一个人。
> 在我把她从你身边带走真正的人。
< on the other hand   dinner and a show certainly would be fun   would n t it  ?
= 但是，吃饭和看演出确实更有意思？
> 另外，晚餐也很有趣，不是吗？
> 另外，晚餐和节目肯定很有趣，不是吗？
> 另外，晚餐和节目肯定会很有趣，不是吗？
> 另外，晚餐会很有趣的，对吧？
> 另外，晚餐和节目肯定会很有趣，对吧？
< but i want our wedding to be what you want it to be
= 但怎样结婚都听你的，
> 但我希望我们的婚礼是你想要的，
> 但我希望我们的婚礼就是你想要的，
> 但我希望我们的婚礼是你们想要的，
> 但是我希望我们的婚礼是你想要的，
> 但我希望我们的婚礼成为你想要的，
< i told you before   i ca n t just give them to you  !
= 我告诉过你，我不能还给你！
> 我告诉过你，我不能就这么给你！
> 我告诉过你，我不能给你！
> 我之前告诉过你，我不能就这么给你！
> 我以前告诉过你，我不能就这么给你！
> 我之前告诉过你，我不能给你！
< i  d go out of my mind  .
= 我想远了。
> 我早就疯了。
> 我早就疯掉了。
> 我就会发疯。
> 我早就失去理智了。
> 我早就发疯了。
< a short one   mr . chairman  .   very well   please proceed  .
= 很短，主席先生。- 很好，请开始吧。
> 很短，主席先生。- 很好，请继续。
> 少一个，主席先生。- 很好，请继续。
> 简单的，主席先生。- 很好，请继续。
> 少一会，主席先生。- 很好，请继续。
> 一小一个，主席先生。- 很好，请继续。
< and i think   you know   we  ll be all right  . no matter what  .
= 我就想，你知道的，无论如何。我们都会好的。
> 我觉得无论如何我们都会没事的。
> 不管怎样，我们都会没事的。
> 我觉得不管怎样我们都会没事的。
> 我觉得无论如何都会好起来的。
> 我觉得不管怎样，我们都会没事的。
< i know what he  s up to  . he wants to try to make the arrest himself  .
= 我知道他来做什么。他想亲自逮捕凶手。
> 我知道他在干什么。他想逮捕自己。
> 我知道他在做什么。他想逮捕自己。
> 我知道他在干什么。他想要逮捕自己。
> 我知道他在做什么。他想要逮捕自己。
> 我知道他在干什么。他想逮捕他自己。
< it broke his heart   but he did n t want to tear apart our family  .
= 他很难过，但他不想破坏我们的家庭。
> 它伤了他的心，但他不想破坏我们的家庭。
> 伤了他的心，但他不想破坏我们的家庭。
> 它伤了他的心，但他不想破坏我们家。
> 它打破了他的心，但他不想破坏我们的家庭。
> 它伤了他的心，但是他不想破坏我们的家庭。
< it  s like knows where we  re going and what we  re thinking  .
= 他好像知道我们将要去哪里以及我们在想什么。
> 就好像知道我们要去哪里。
> 就像知道我们要去哪里。
> 就像知道我们要去哪里想什么。
> 好像知道我们要去哪里想什么。
> 就像是知道我们要去哪里想什么。
< because there  s still a chance that you might be alive tomorrow
= 因为你还有希望可能明天还活着，
> 因为还有可能你明天还活着，
> 因为你还有可能明天还活着，
> 因为还是有可能你明天还活着，
> 因为仍然有可能你明天还活着，
> 因为仍有可能你明天还活着，
< i could see the change in him when his dad came back  .
= 我可以看出他爸爸回来时他变了。
> 我可以看到他爸爸回来的改变。
> 我可以看到他父亲回来的改变。
> 我可以看到他爸爸回来后的改变。
> 我可以看到他爸爸回来时的改变。
> 我可以看到他爸爸回来时的变化。
< noticed since the return of the boy to his mother  .
= 他提到自从孩子回到他母亲身边后。
> 我就注意到了。
> 从孩子回来后就注意到了。
> 从他母亲回来后就注意到了。
> 从他母亲身边就注意到了。
> 从他母亲身边注意到。
< that  s why she brought me along   but honestly i  m not interested   no offense  .
= 所以才带我来，不过说实话，我没有兴趣，不要介意。
> 所以她带我来，但老实说我没兴趣，无意冒犯。
> 所以她带我来，但老实说，我没兴趣。
> 所以她带我来，但老实说，我没兴趣，无意冒犯。
> 所以她带我一起来，但老实说，我没兴趣。
> 所以她带我来，但老实说，我没有兴趣，无意冒犯。
< and i like your dress   which means i can see  .
= 我喜欢你的衣服，说明我能看见。
> 我喜欢你的衣服，这意味着我能看见。
> 我喜欢你的衣服，这意味着我能看到。
> 我喜欢你的衣服，我可以看到。
> 我喜欢你的裙子，这意味着我能看见。
> 我喜欢你的裙子，这意味着我能看到。
< well   i would n t go so far as to call them friendly
= 呃，我也不会说多么的友好，
> 我可不会把他们叫做友好，
> 我可不会把他们叫友好，
> 嗯，我不会把他们叫做友好，
> 嗯，我不会把他们叫友好，
> 好吧，我不会把他们叫做友好，
< sorry we  re late   we had to review the research budget  .
= 抱歉，我们迟到了 - 我们得重新评估研究预算。
> 抱歉，我们迟到了，我们得检查研究预算。
> 抱歉，我们迟到了，我们得检查研究预算预算。
> 抱歉，我们迟到了，我们得检查一下研究预算。
> 抱歉，我们迟到了，我们得检查一下调查预算。
> 抱歉，我们迟到了，我们必须检查研究预算。
< if you want to remain team manager   you have to do this  .
= 如果你还想做球队经理的话你必须得做这个。
> 如果你想继续当球队经理，你必须这么做。
> 如果你想继续当团队经理，你必须这么做。
> 如果你还想当球队经理，你必须这么做。
> 如果你还想当团队经理，你必须这么做。
> 如果你想继续当球队经理，你就必须这么做。
< i had a really nice time on our date tonight  .
= 今晚的约会很开心。
> 今晚我们的约会很愉快。
> 今晚我们的约会真的很愉快。
> 今晚我们约会很愉快。
> 今晚我们的约会真的很开心。
> 我今晚约会很愉快。
< we do n t even have to be a couple at all   even   you know   but   um
= 我们甚至可以根本不成为夫妻，你知道，但是，嗯，
> 我们甚至不用成为一对，你知道，
> 我们甚至不用成为一对，你知道的，
> 我们甚至不需要一对夫妇，你知道，
> 我们甚至不需要一对，你知道的，
> 我们甚至不用成为一对，你知道，但是，
< you  ll be a father for as long as you can  .
= 你将成为一位父亲能做多久做多久。
> 你将尽可能成为父亲。
> 你将尽可能成为一个父亲。
> 你将尽可能做父亲。
> 你要尽可能成为父亲。
> 你要尽可能做父亲。
< because i am done with her   and for real this time  .
= 因为我和她完了，这次是真的。
> 因为我受够了，这次真的。
> 因为我受够了，这次是真的。
> 因为我已经受够了，这次真的。
> 因为我和她完了，这次是真的。
> 因为我和她结束了，这次是真的。
< the hospital says you  re gon na need somebody to live in the house with you
= 医院说你需要有人住在你家里照顾你，
> 医院说你需要有人和你住一起，
> 医院说你需要有人跟你住一起，
> 医院说你需要有人和你一起住
> 医院说你需要有人和你一起住，
> 医院说你需要有人跟你一起住
< and on the night he died the last thing he said to her was  .
= 他死的那晚对她所说的最后一句话是。
> 昨晚他死的最后一句话就是。
> 昨晚他死的最后一句话是。
> 他死于他对她说的最后一句话。
> 他死的那天晚上他对她说的最后一句话是。
> 他死的那天晚上他对她说的最后一句话就是。
< i know i  m going to go back home and cry
= 我知道我得回家了我会哭的，
> 我知道我要回家哭泣，
> 我知道我要回家哭了，
> 我知道我要回家哭，
> 我知道我会回家哭泣，
> 我知道我要回家哭泣 -
< i know you do   but i  m here to tell you the bad news  .
= 我明白，不过我来是有坏消息要告诉你。
> 我知道，但我是来告诉你坏消息的。
> 我知道，但是我是来告诉你坏消息的。
> 我知道，可我是来告诉你坏消息的。
> 我知道，但我是来告诉你这个坏消息的。
> 我知道你想，但我是来告诉你坏消息的。
< like you  d expect   i guess  . i feel sorry for her  .
= 就像你所想像的，我猜。我为她感到难过。
> 像你期望的那样，我想。我为她感到难过。
> 像你希望的那样，我想。我为她感到难过。
> 像你期望的那样，我想。我为她难过。
> 就像你期望的，我想。我为她感到难过。
> 像你希望的那样，我想。我为她难过。
< i know   but if you think about it in practical terms
= 我知道，但如果你想的实际一点，
> 我知道，但如果你仔细想想的话，
> 我知道，但如果你仔细想想，
> 我知道，但如果你从实际角度想，
> 我知道，但如果你仔细考虑的话，
> 我知道，但如果你从实际角度想想，
< i hope for my father  s sake this is one of those times  .
= 为了我父亲希望这次是大多数中的一次。
> 希望我父亲是这样的时刻。
> 但愿我父亲是这样的时刻。
> 我也希望我父亲是这样的时刻。
> 我也希望我父亲这次是这样。
> 为了我父亲我希望这次是这样。
< so i could n t expect she would let me stay home
= 所以我估计她不会让我呆在家里，
> 所以我不能指望她会让我在家，
> 所以我不指望她会让我在家，
> 所以我不能指望她会让我呆在家，
> 所以我不指望她会让我呆在家，
> 所以我不能指望她会让我留在家里，
< what will l have to do to make you say those words  ?
= 你要我怎么做才会对我说这3个字？
> 我要怎么做才能让你说这些话？
> 我要怎么做才能让你说那些话？
> 我要怎么做才能让你说？
> 我该怎么做才能让你说这些话？
> 我该怎么做才能让你说那些话？
< we should have done something about that guy months ago  .
= 几个月前我们就该对他做点什么了。
> 几个月前我们就应该做点什么。
> 几个月前我们应该做点什么。
> 几个月前我们就该做点什么。
> 几个月前我们就该对那家伙做点什么。
> 几个月前我们就该对他做点什么。
< oh   god   i  m sorry  . before i knew it was you
= 噢，天呐，抱歉。我之前还不知道是你时，
> 哦，天哪，对不起。在我知道是你之前，
> 哦，天啊，对不起。在我知道是你之前，
> 天啊，抱歉。在我知道是你之前，
> 天啊，对不起。在我知道是你之前，
> 天哪，对不起。在我知道是你之前，
< all right   you  re alive  !   great to hear your voices again
= 太棒了，你还活着！- 很高兴再次听到你们的声音
> 好吧，你还活着！- 很高兴再次听到你的声音
> 好吧，你还活着！- 很高兴又听到你的声音
> 好吧，你还活着！- 再次听到你的声音
> 好的，你还活着！—很高兴再次听到你的声音
> 好的，你还活着！—再次听到你的声音
< see if you can find another match  . let me know  . all right  .
= 如果有类似发现，记得向我报告。好的。
> 再找一个匹配的。告诉我。好的。
> 再找一个匹配的。告诉我。好吧。
> 再找个匹配的。告诉我。好的。
> 再找一个匹配的。告诉我。好。
> 再找个匹配的。告诉我。好吧。
< i  ve been out there for   minutes   yelling for help  .
= 我在外面，喊了快20分钟。
> 我在外面呆了20分钟，喊人帮忙。
> 我在外面呆了15分钟，喊人帮忙。
> 我在外面呆了10分钟，喊人帮忙。
> 我在外面等了20分钟，喊人帮忙。
> 我在外面待了20分钟，喊人帮忙。
< i was thinking that you might want to offer her
= 我在想，你也许应该
> 我觉得你可能想给她
> 我在想你可能想给她
> 我在想，也许你想给她
> 我在想，你是不是想给她
> 我在想，你可能想给她
< for a long time   this place and those guys have been my whole world  .
= 在过去的很长时间里，这个地方还有他们就是我的全部。
> 很长一段时间，这里和那些人都是我的全部。
> 很长一段时间，这个地方和那些人都是我的全部。
> 很长一段时间，这地方和那些人都是我的全部。
> 很长一段时间，这个地方和他们是我的全部。
> 很长一段时间，这里和他们是我的全部。
< i told her she might have to get her own place
= 我告诉她她应该自己找地方住 ，
> 我告诉她她可能要自己找个地方，
> 我告诉她她可能要找个自己的地方，
> 我告诉她她可能要自己找地方，
> 我告诉她她可能得找个自己的地方，
> 我告诉她她可能得自己找个地方，
< you remember my brother   the one who lives in brazil  ?
= 你还记得我弟弟吗，住在巴西的那个？
> 你记得我弟弟，住在巴西的那个吗？
> 你记得我哥哥，住在巴西的那个吗？
> 你还记得我弟弟住在巴西的那个吗？
> 你还记得我哥哥住在巴西的那个吗？
> 你记得我弟弟，住在巴西的那个？
< the only reason that you  re even talking about a house is to make me feel bad  .
= 你跟我谈房子就是让我难受。
> 你说房子就是让我难过。
> 你说房子只是让我难过。
> 你说房子的唯一原因就是让我难过。
> 你说房子就是让我很难过。
> 你说房子就是想让我难过。
< i do n t wan na hear about it  .   you will hear about it  !
= 我不想听。-你要听！
> 我不想听。- 你会听到的！
> 我不想听。- 你会知道的！
> 我不想听这些。- 你会听到的！
> 我不想听这个。- 你会听到的！
> 我不想听。- 你会听见的！


</pre>