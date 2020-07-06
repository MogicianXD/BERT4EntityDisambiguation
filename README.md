# BERT4EntityDisambiguation
use BERT in pytorch for Entity Disambiguation task

# 任务描述
现有douban、mtime、maoyan三个来源的电影，包含名称、简介、导演、演员、类型等等属性。
需要相同的电影融合为一个电影条目，其中maoyan数量很少，可以合并到mtime中。

参照实体消歧（或实体链接）的惯用语，数量最多的douban可以作为知识库中的entity，mtime（+maoyan）作为需要链接到KB的mention。

# 思路
事实上，这个任务也相当于文本多分类任务。类别数为KB中的entity数，那么将模型运算得到的mention hidden states输入到最后一层全连接层，哪个位置的神经元输出最高，即意味着属于该index对应类。

但是，KB中的entity数量并不是固定不变的。如果有新的entity加入，那么需要重新训练新维度的linear layer。因此，不能使用这样的池化操作来进行分类。

这个问题也可以通过计算相似度，然后取最高的来完成。

通过模型（这里使用单纯的BERT）计算得到每个mention的emb和每个entity的emb，二者计算得到余弦相似度（或者计算点积距离，不必归一化，实验中采取这种方式），再取相似度最高的作为预测结果。

# 架构
模型主要使用BERT来计算文本相似度，电影名和简介作为两个输入，实际会拼接为 [CLS]电影名[SEP]简介[SEP]。当然也可以使用RoBERT等transformer模型。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200702100032824.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MjEyNzE4Mg==,size_16,color_FFFFFF,t_70)
$M = BERT(text_M)$
$E = BERT(text_E)$

得到的mention的embs $M$ 与entity的embs $E$ 做点积，再通过softmax激活，得到评分矩阵$S$，$S_{ij}$即表示$M_i$与$E_j$是同一个实体的概率。

$S = \sigma{(ME^T)}$

sigmoid适用于二分类，softmax则适用于多分类任务。这里使用余弦相似度也未尝不可，一样能够得到小于1的scores。但是观察发现，余弦相似度最低也是0.5，相似度0.9以上的候选项非常之多，很难找到一个合适的阈值过滤无效预测。
另外，尽管KB中可能本来就有重复的entity，但这样的数量应当非常之少，绝大多数mention对应了小于等于1个entity，因此可以视作多分类任务，而非多标签任务。

损失函数使用交叉熵损失函数

$L = CrossEntropyLoss(S, groundtruth)$

# 实验
数据集包括完整的属性信息details.csv和真实的链接映射link.csv，按照惯例，以6:2:2分割link.csv。

与我此前做过的实验相比，特殊的是，train、valid、test三个过程差别很大。
## Train
将mention和对应实体的文本成对地存入Dataloader。按batch取到一个mention_batch和entity_batch，分别送入模型计算得到$M$、$E$，然后计算损失，进行反向传播。

注：如果采用余弦相似度（我最开始的做法），没必要依前述对二者做矩阵乘，在计算损失时，事实上只用到了groundtruth对应的概率。因此，只需要按行求二者行向量的余弦相似度即可。

## Valid
输入为mention对应的文本，以及所有entity的文本，形成两个DataLoader。但是将所有entity都计算一遍耗时过长，因此只计算mention对应groundtruth的文本集合。

### 指标
多分类任务也可以使用二分类任务常用的precision、recall、F1、AUC。但在平均上，有macro和micro之分。

如上述所说，一个entity对应的mention一般为0或1个。换句话说，类别样本比较均衡。因此使用不考虑数据数量，平等看待每一类的macro-F1即可。

实验中，我使用的Acc最高可达**98%**，AUC最高可达**99.99%**，已经相当之高了，因此没有横向的比较实验。

## Test
因为只在Valid最佳时进行Test（Valid指标最佳时保存模型），次数少，因此这次的输入就应当是所有entity了。

因为数据量增加，Acc指标有所下降，但也是96%

现在需要找到一个阈值，概率大于的算有效预测，小于的不放入预测结果中，首先保证有效预测的尽可能正确（precison），其次最好预测出来的足够多（recall），这个和PR曲线的阈值不大一样，需要自己手写。

另外，既然侧重于精度precision，那么可以用$\beta$较小（我取的0.8）的$F_\beta score$来衡量。

以0.01为步长，遍历0.70到0.99之间的可能阈值，precision<0.9直接不考虑，然后找什么时候F-score最大，结果就是0.7...

# classify
然后对数据库中尚未确定的电影进行分类。

输出pred.csv，第一列为douban _id，第二列为other _id，第三列为概率时。共5w+条，随机选取样例发现0.7以上比较准确，以0.75为阈值筛出1.2w，以0.7为阈值筛出1.36w.

# 还可以改进的地方
首先，因为选取的数据集电影名称一定是相同的，虽然这符合正常的模式，但是还是有可能导致模型过分倾向于名称，简介影响变小。因此，可以加入负样本，最好为电影名相同，但并非同一电影的负样本，不过可能数据量比较少，还需要其他负样本填补。然后选择一个pairwise的loss进行优化。

其次，BERT可以接受一到两句话作为输入，并希望每句话的开头和结尾都有特殊的标记。（我使用的是[transformers](https://github.com/huggingface/transformers)提供的pytorch实现）。所以我只输入了两个属性，其他属性没有用上。

其实也可以用另一个BERT来计算其他属性提供的hidden state，与之前的做一个多模态的串联，加一个attention层输出得到最终的emb表示。

导演、演员、类型为列表形式，可以直接输入字符串（但可能没有多大效果），也可以用GRU来计算。因为一般排在最前面的最重要，所以可以逆序输入。

另外，可以用TF-IDF、fastText等模型来比较我们模型的性能。因为时间不够，所以不考虑了。

