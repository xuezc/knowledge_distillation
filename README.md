# knowledge_distillation

一．从类概率蒸馏
1.	QUEST: Quantized embedding space for transferring knowledge(CVPR 2020)：代替在教师网络的初始特征空间进行蒸馏操作，文章先将原始特征空间转化为一个对特征扰动更为鲁棒的量化空间再进行蒸馏。在该量化空间中，更关注重要的语义概念及其在知识蒸馏中的空间相关性。具体的网络结构图如下图所示
![paper1](https://github.com/xuezc/knowledge_distillation/blob/master/image1.png)
    &emsp;在该方法中，首先，学习一个预定义的词汇表的教师深层特征（称之为视觉教师词汇）,具体操作是先定义一个视觉词嵌入的词汇表V，并取最后一个隐层的特征图f_T，以此使用平方欧几里得距离来计算特征图和视觉词的距离d
![paper1](https://github.com/xuezc/knowledge_distillation/blob/master/image2.png)    

    接着使用计算出来的距离计算软分配向量P_T，如下式所示，其中tau表示温度系数，用以控制分配的软化度
![paper1](https://github.com/xuezc/knowledge_distillation/blob/master/image3.png)

    然后，使用分配预测器由学生网络来预测向量P_S，其中分配预测器是由一个基于余弦相似度的卷积层组成的，计算公式如下式，其中，W表示卷积层的参数，
![paper1](https://github.com/xuezc/knowledge_distillation/blob/master/image4.png)

    最后就可以计算向量P_T和P_S的KL散度
![paper1](https://github.com/xuezc/knowledge_distillation/blob/master/image5.png)

    这样的蒸馏策略旨在通过预先计算好的教师词典进行预测，以使学生网络行为与教师网络行为相一致，而不是使用特征预测。这种方法的优势在于，它只对教师在培训过程中学习到的主要视觉概念/单词进行编码，而对教师特征的扰动不敏感。
2.	Ensemble Distribution Distillation(ICLR 2020)：文章将集合蒸馏和先验网络相结合，提出了一种新的集合分布蒸馏方法，将集合分布蒸馏到一个先验网络中，这使得单个模型既能保留改进的分类性能，又能保留集合的多样性。文章采用了贝叶斯方法的集合，因为这样可以使知识不确定性和贝叶斯模型不确定性联系起来。同样的，在计算损失时，文章也使用了KL散度。与其他方法不同的是使用了模型集合，以此能够让学生网络去学习模型集合的平均，使得其分类精度更加准确。此外，文章还引入了先验网络，即使用了一个简单的网络来对输出分布参数化为条件分布，使得先验网络能够有效地模仿一个集合。
    对于一个集合进行蒸馏，一般使用最小化模型和集合的预测分布的KL散度，如下式
![paper2](https://github.com/xuezc/knowledge_distillation/blob/master/image6.png)

    但是，这样会损失掉模型集合的多样性。而文章提出的Ensemble Distribution Distillation方法就旨在利用损失掉的多样性，这主要是通过引入先验网络，先验网络通过参数化Dirichlet分布来模拟分类输出分布上的分布
![paper2](https://github.com/xuezc/knowledge_distillation/blob/master/image7.png)

    对于给定的转移集合，先验网络通过最小化每个分类分布的负对数似然来训练
![paper2](https://github.com/xuezc/knowledge_distillation/blob/master/image8.png)

    因此，集合分布蒸馏是对先验网络模型的最大似然估计的直接应用。给定一个蒸馏先验网络的分布，预测分布由Dirichlet先验下的期望分类分布给出：
![paper2](https://github.com/xuezc/knowledge_distillation/blob/master/image9.png)

    进而通过考虑预测y与分类参数之间的相互信息，可以得到不确定性的可分离测度
![paper2](https://github.com/xuezc/knowledge_distillation/blob/master/image10.png)

    上式允许将由期望分布的熵给出的总不确定性分解为数据不确定性和知识不确定性。如果集合分布蒸馏完全成功，则从分布蒸馏先验网络导出的不确定性度量应与从原始集合导出的度量相同。
3.	Noisy Collaboration in Knowledge Distillation(ICLR 2020)：文章认为噪声是改进神经网络训练和解决明显矛盾的目标的一个关键因素，即可以提高模型的泛化性和鲁棒性。受多个噪声源引起的大脑变异性试验的启发，通过输入电平或监测信号的噪声来引入变异性。结果表明，噪声可以提高模型的泛化能力和鲁棒性。
    具体地，文章试验了加入多种不同类型和不同数量的噪声来测试噪声对知识蒸馏的影响，在蒸馏策略中，文章使用了以下的损失函数，S()表示学生网络，delta表示加入的噪声。
![paper3](https://github.com/xuezc/knowledge_distillation/blob/master/image11.png)


二．从中间层蒸馏
1.	Feature-map-level Online Adversarial Knowledge Distillation(ICLR 2020)：文章提出一种在线蒸馏方法，该方法同时训练多个网络，并通过使用判别器来区分不同网络的特征图分布。其中，每个网络都有相应的判别器，该判别器在将另一个网络的特征图分类为真的同时，将特征图与自身的特征图区分为假。通过训练一个网络来欺骗相应的判别器，它可以学习另一个网络的特征图分布。此外，文章还提出了一种循环学习方法来训练两个以上的网络。文章将该方法应用到分类任务的各种网络结构中，发现在训练一对小网络和一对大网络的情况下，性能有显著的提高。
    文章提出的方法称为Online Adversarial Feature map Distillation (AFD)，网络的结构图如下图所示
![paper4](https://github.com/xuezc/knowledge_distillation/blob/master/image12.png)  

    在线训练两个不同的网络network1和2时，使用了两个鉴别器D1和D2。在训练D1时，使D2的特征图被视为真，而D1的特征图被视为假，反之亦然。然后，训练每个网络D1和D2欺骗其对应的鉴别器，使其能够生成模仿另一个网络的特征图的特征图。在整个对抗训练过程中，每个网络学习另一个网络的特征图分布。通过同时利用基于logit的蒸馏损失和基于特征图的对抗损失，可以观察到不同网络结构对性能的显著改善，特别是在一起训练小型和大型网络时。
    对于通常的相互学习网络，两个网络的总体损失函数为
![paper4](https://github.com/xuezc/knowledge_distillation/blob/master/image13.png)

    文章提出的AFD法使用对抗性训练在特征图级别传递知识。AFD将网络分为两部分，一部分是生成特征图的特征提取部分，另一部分是将特征图转换为logit的分类器部分。每个网络还具有相应的判别器，用于区分不同的特征图分布。在此，将特征提取部分命名为Gk，其鉴别部分命名为Dk，k表示网络号。每个网络必须欺骗其鉴别器以模拟对等网络的特征图，鉴别器必须区分特征图来自哪个网络。鉴别器和特征提取器的总体对抗损失如下
    ![paper4](https://github.com/xuezc/knowledge_distillation/blob/master/image14.png)
    
    特征提取器G1和G2接受输入x并生成特征图。鉴别器D1获取特征图并产生介于0（假）和1（实）之间的标量。如果特征图来自共同训练的网络（在本例中为G2），则训练输出1；如果特征图来自其所属的网络（在本例中为G1），则训练输出0。D1的目标是通过正确区分两种不同的特征图分布来最小化鉴别器损失项LD1，G1的目标是通过欺骗D1使损失项LG1最小化，从而错误地将G1的特征图确定为实数并得到1。每个训练网络的目标是最小化LGk，以模拟对等网络的特征图分布。通过改变两个网络的角色，这种对抗性方案的工作原理完全相同。
    通过结合基于logit的损失和基于对抗特征地图的损失，网络1和2的总体损失如下
    ![paper4](https://github.com/xuezc/knowledge_distillation/blob/master/image15.png)
    
    需要注意的是，基于logit的损失项L_logit和基于特征映射的损失项L_G并不是由同一个优化器优化的。实际上，它们是在同一个小批量中交替优化的。在每一次小批量迭代中，将一幅图像推导出一个模型，然后计算一个logit和一个特征图。然后分别计算这两个损失项，并基于这两个损失对网络进行优化，即用基于logit的损失更新参数一次，然后用基于特征图的损失重新更新参数。其中为每个损失项分别优化的原因是它们分别使用了不同的学习率。对抗性损失需要较小的学习率，因此如果使用相同的优化器和相同的学习速率，网络将不会被优化。
    文章还提出了一种同时训练两个以上网络的循环学习方案。它将所需的鉴别器的数目减少到K，其中K是参与的网络的数目。这种循环学习框架不仅比双向学习方法需要更少的计算量，而且与其他多网络在线训练方案相比，可以取得更好的效果。具体的结构图如下图所示
    ![paper4](https://github.com/xuezc/knowledge_distillation/blob/master/image16.png)
    
    在网络中，每个网络以单向循环的方式将其知识传输到下一个对等网络。如果一起训练K个网络，每个网络将其知识提取到下一个网络，除了最后一个网络将其知识转移到第一个网络之外，创建一个循环的知识转移流为1>2,2>3 ,…,(K-1)>K,K>1。使用这种循环学习框架的主要好处是避免使用过多的鉴别器。如果对每一对网络应用提出的对抗损失，它将需要两倍于每一对可能的K网络的数量，这将需要大量的计算。经验表明，对于多个网络，循环训练方案优于其他在线方法的训练方案。
2.	Knowledge Squeezed Adversarial Network Compression(AAAI 2020)：受网络规模差距过大，小网络无法完美模拟大网络的假设启发，文章提出了一种在对抗性训练框架下学习学生网络的知识转移方法，包括有效的中间监督。为了实现功能强大和高度紧凑的中间信息表示，压缩的知识通过任务驱动的注意机制来实现。这样，教师网络的知识转移就可以适应学生网络的规模。结果表明，该方法综合了面向过程学习和面向结果学习的优点。
3.	Knowledge Distillation from Internal Representations(AAAI 2020)：文章指出了当教师规模相当大时，并不能保证教师的知识会转移到学生身上；即使学生与软标签紧密匹配，其内部表现也可能大不相同。这种内部的不匹配可能会破坏最初从教师转移到学生身上的泛化能力。文中除了使用kl散度之外，使用了余弦相似度损失用以转移中间层的暗知识。
4.	Structured Knowledge Distillation for Dense Prediction：文章将结构信息从大型网络传输到小型网络来进行密集预测任务，文章具体研究了两种结构化的蒸馏方案，一是通过建立静态图来提取成对相似度的成对蒸馏，另一个是使用对抗训练来提取整体知识的整体蒸馏。并通过对语义分割、深度估计和目标检测这三个密集预测任务的大量实验，来证明提出方法的有效性。
