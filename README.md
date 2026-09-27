# Bert

（1）解决的问题

传统语言模型是单向的（transformer、elmo），单向编码存在局限，问答问题的答案通常和双向内容相关。Bert通过掩码解决了传统语言模型单向的问题。

（2）Bert结构

Bert-Base：12层 768维 12头 和GPT规模一致好做对比

Bert-Large：24层 1024维 16头

[CLS]/[SEL]/token embedding/segment embedding/position embedding

（3）预训练和微调

- 预训练：
    

```
MLM（掩码实现双向信息编码 15%（80% 10% 10%））、NSP（学习句子和句子之间的内容信息）
```

- 微调
    

```
文本分类使用[CLS]微调，并且微调时模型结构几乎不需要改变
```

      BERT预训练后保留Embedding和Transformer Encoder等主体参数，通常移除MLM、NSP任务头。面对不同下游任务，主要改变输入组织方式并增加任务特定输出层。标准微调通过下游任务损失进行端到端反向传播，同时更新任务输出层和BERT主体参数。

**关于bert的一些问题**

bert的研究动机和技术位置

（1）Elmo为什么不是真正的深度双向建模（深度双向：左右信息在每一层内部深度建模）

Elmo用LSTM实现了前向和后向两个方向的建模编码，但***两个方向在语言模型内部独立计算，左右信息没有在每一层中联合交互***，因此BERT论文将其视为浅层拼接式双向建模。

（2）Transformer Encoder为什么能实现双向编码

Transformer注意力计算公式：

$$
\begin{align\*}
&\text{Attention}(Q,K,V) = \text{Softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\end{align\*}

$$

编码器Encoder中每个token都有自己的Q K V，***不使用Causal mask使得每个token的Q都可以关注到句子中所有有效的K***，即编码器每个token可以看到整条句子中完整的所有token，所以每一层encoder都能联合融合左右信息。

（3）Bert为什么使用transformer encoder不是decoder

transformer的encoder可以利用token左右上下文适合理解和编码，decoder有causal mask只能用当前位置和当前位置之前的位置，适合自回归生成。

BERT主要解决语言理解和表示学习问题。需要为每个token同时融合左右上下文的表示，因此继承了transformer中编码器encoder的方法，并在此基础上实现了MLM和NSP，为token编码结果获取更多信息。

（4）无Causal Mask的encoder和MLM分别发挥什么作用

无Causal Mask的Encoder是transformer的编码器用于为token学习编码向量，为模型提供了让一个token同时关注左右所有token的能力。MLM是训练任务，遮住答案迫使模型利用左右上下文学习语言规律。

所以，Transformer Encoder提供”双向建模能力“，MLM提供”训练这种双向建模能力的子监督目标“。

（5）Bert如何形成“预训练-微调”范式

Bert先通过数据做模型的无监督预训练，通过预训练得到模型参数。从结构上讲微调只需要在预训练结构的基础上根据微调任务特点为模型增加输出层即可实现利用任务损失对BERT主体和任务头进行端到端微调，从而减少从零设计和训练专用模型的成本。

（6）Transformer、Elmo、Bert之间的关系

Elmo通过LSTM前向/后向编码实现了简单的双向编码，为一个词根据不同上下文生成不同表示。完成训练后保存双向语言模型参数，并根据输入句子的上下文动态计算词向量。但是，Elmo两个方向在模型内部单独计算，通常只在输出时才进行拼接，因此属于浅层双向建模，不是在每一层中融合左右的信息。

Transformer由encoder和decoder组成。Encoder中的self-attention中没有causal mask，每个token可以同时关注上下文，实现联合双向编码。Decoder中的Masked self-attention 使用casual mask只允许当前位置关注自己以及之前的位置，用于自回归生成。

Bert继承Transformer的Encoder，通过MLM遮盖部分token，使模型不能直接看到预测目标，让模型必须利用左右上下文恢复原词。由于左右信息会在多层Encoder中反复交互，因此可以实现深度双向编码。预训练完成后，BERT可以通过增加较小的任务输出层对模型参数进行微调来适配文本分类、序列标注、问答等下游任务，减少从零设计和训练任务模型的成本。

MLM和NSP

（1）15%的Token如何选取

首先，根据attention_mask（确定token不是[pad]）和special_mask（不是特殊[CLS][SEP]）确定句子中可以被选取的token位置。然后按照0.15的概率通过伯努利随机选取选择token。

选中的token80%替换为[MASK]，10%替换为随机token，10%不变保留原token。

（2）为什么MLM labels未预测位置设为-100

\-100用于提醒模型当前位置不参与损失计算。

（3）MLM输出为什么是【B,L,vocab_size】

B表示batch_size，L表示当前句子的序列长度包括特殊token和padding，vocab_size表示每个token位置都有vocab_size个单词作为可能填到该位置的token。

（4）NSP的正负样本如何构造

两个句子之间存在前后关系，则标记NSP label为0；如果两个句子之间不存在前后关系则标记NSP label为1。

（5）MLM和NSP损失如何同时训练BERT

Bert结构先是encode然后分两支MLM和NSP，通过损失反向传播训练模型时，MLM Head只接收MLM损失的梯度，NSP Head只接收NSP损失的梯度，共享Encoder同时接收两个损失的梯度之和，一次optimizer.step()同一更新所有参数。

（6）NSP后来为什么被RoBERTa删除

RoBERTa通过消融实验发现，在其训练配置下移除NSP，模型性能可以持平或提高，说明原始NSP并不是获得强语言表示必不可少的预训练目标。
