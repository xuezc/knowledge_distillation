# knowledge_distillation

## 一．从类概率蒸馏
1.	QUEST: Quantized embedding space for transferring knowledge(CVPR 2020)：代替在教师网络的初始特征空间进行蒸馏操作，文章先将原始特征空间转化为一个对特征扰动更为鲁棒的量化空间再进行蒸馏。在该量化空间中，更关注重要的语义概念及其在知识蒸馏中的空间相关性。具体的网络结构图如下图所示

![paper1](https://github.com/xuezc/knowledge_distillation/blob/master/images/images/image1.png)
    
&emsp;&emsp;在该方法中，首先，学习一个预定义的词汇表的教师深层特征（称之为视觉教师词汇）,具体操作是先定义一个视觉词嵌入的词汇表V，并取最后一个隐层的特征图f_T，以此使用平方欧几里得距离来计算特征图和视觉词的距离d

![paper1](https://github.com/xuezc/knowledge_distillation/blob/master/images/image2.png)    

&emsp;&emsp;接着使用计算出来的距离计算软分配向量P_T，如下式所示，其中tau表示温度系数，用以控制分配的软化度

![paper1](https://github.com/xuezc/knowledge_distillation/blob/master/images/image3.png)

&emsp;&emsp;然后，使用分配预测器由学生网络来预测向量P_S，其中分配预测器是由一个基于余弦相似度的卷积层组成的，计算公式如下式，其中，W表示卷积层的参数，

![paper1](https://github.com/xuezc/knowledge_distillation/blob/master/images/image4.png)

&emsp;&emsp;最后就可以计算向量P_T和P_S的KL散度

![paper1](https://github.com/xuezc/knowledge_distillation/blob/master/images/image5.png)
    
&emsp;&emsp;这样的蒸馏策略旨在通过预先计算好的教师词典进行预测，以使学生网络行为与教师网络行为相一致，而不是使用特征预测。这种方法的优势在于，它只对教师在培训过程中学习到的主要视觉概念/单词进行编码，而对教师特征的扰动不敏感。
2.	Ensemble Distribution Distillation(ICLR 2020)：文章将集合蒸馏和先验网络相结合，提出了一种新的集合分布蒸馏方法，将集合分布蒸馏到一个先验网络中，这使得单个模型既能保留改进的分类性能，又能保留集合的多样性。文章采用了贝叶斯方法的集合，因为这样可以使知识不确定性和贝叶斯模型不确定性联系起来。同样的，在计算损失时，文章也使用了KL散度。与其他方法不同的是使用了模型集合，以此能够让学生网络去学习模型集合的平均，使得其分类精度更加准确。此外，文章还引入了先验网络，即使用了一个简单的网络来对输出分布参数化为条件分布，使得先验网络能够有效地模仿一个集合。
    
&emsp;&emsp;对于一个集合进行蒸馏，一般使用最小化模型和集合的预测分布的KL散度，如下式

![paper2](https://github.com/xuezc/knowledge_distillation/blob/master/images/image6.png)  

&emsp;&emsp;但是，这样会损失掉模型集合的多样性。而文章提出的Ensemble Distribution Distillation方法就旨在利用损失掉的多样性，这主要是通过引入先验网络，先验网络通过参数化Dirichlet分布来模拟分类输出分布上的分布

![paper2](https://github.com/xuezc/knowledge_distillation/blob/master/images/image7.png)  

&emsp;&emsp;对于给定的转移集合，先验网络通过最小化每个分类分布的负对数似然来训练

![paper2](https://github.com/xuezc/knowledge_distillation/blob/master/images/image8.png)  
 
&emsp;&emsp;因此，集合分布蒸馏是对先验网络模型的最大似然估计的直接应用。给定一个蒸馏先验网络的分布，预测分布由Dirichlet先验下的期望分类分布给出：

![paper2](https://github.com/xuezc/knowledge_distillation/blob/master/images/image9.png)  

&emsp;&emsp;进而通过考虑预测y与分类参数之间的相互信息，可以得到不确定性的可分离测度

![paper2](https://github.com/xuezc/knowledge_distillation/blob/master/images/image10.png)  

&emsp;&emsp;上式允许将由期望分布的熵给出的总不确定性分解为数据不确定性和知识不确定性。如果集合分布蒸馏完全成功，则从分布蒸馏先验网络导出的不确定性度量应与从原始集合导出的度量相同。
3.	Noisy Collaboration in Knowledge Distillation(ICLR 2020)：文章认为噪声是改进神经网络训练和解决明显矛盾的目标的一个关键因素，即可以提高模型的泛化性和鲁棒性。受多个噪声源引起的大脑变异性试验的启发，通过输入电平或监测信号的噪声来引入变异性。结果表明，噪声可以提高模型的泛化能力和鲁棒性。
    
&emsp;&emsp;具体地，文章试验了加入多种不同类型和不同数量的噪声来测试噪声对知识蒸馏的影响，在蒸馏策略中，文章使用了以下的损失函数，S()表示学生网络，delta表示加入的噪声。
![paper3](https://github.com/xuezc/knowledge_distillation/blob/master/images/image11.png)


## 二．从中间层蒸馏
1.	Feature-map-level Online Adversarial Knowledge Distillation(ICLR 2020)：文章提出一种在线蒸馏方法，该方法同时训练多个网络，并通过使用判别器来区分不同网络的特征图分布。其中，每个网络都有相应的判别器，该判别器在将另一个网络的特征图分类为真的同时，将特征图与自身的特征图区分为假。通过训练一个网络来欺骗相应的判别器，它可以学习另一个网络的特征图分布。此外，文章还提出了一种循环学习方法来训练两个以上的网络。文章将该方法应用到分类任务的各种网络结构中，发现在训练一对小网络和一对大网络的情况下，性能有显著的提高。
    
&emsp;&emsp;文章提出的方法称为Online Adversarial Feature map Distillation (AFD)，网络的结构图如下图所示

![paper4](https://github.com/xuezc/knowledge_distillation/blob/master/images/image12.png)  
    
&emsp;&emsp;在线训练两个不同的网络network1和2时，使用了两个鉴别器D1和D2。在训练D1时，使D2的特征图被视为真，而D1的特征图被视为假，反之亦然。然后，训练每个网络D1和D2欺骗其对应的鉴别器，使其能够生成模仿另一个网络的特征图的特征图。在整个对抗训练过程中，每个网络学习另一个网络的特征图分布。通过同时利用基于logit的蒸馏损失和基于特征图的对抗损失，可以观察到不同网络结构对性能的显著改善，特别是在一起训练小型和大型网络时。
    
&emsp;&emsp;对于通常的相互学习网络，两个网络的总体损失函数为

![paper4](https://github.com/xuezc/knowledge_distillation/blob/master/images/image13.png)  
    
&emsp;&emsp;文章提出的AFD法使用对抗性训练在特征图级别传递知识。AFD将网络分为两部分，一部分是生成特征图的特征提取部分，另一部分是将特征图转换为logit的分类器部分。每个网络还具有相应的判别器，用于区分不同的特征图分布。在此，将特征提取部分命名为Gk，其鉴别部分命名为Dk，k表示网络号。每个网络必须欺骗其鉴别器以模拟对等网络的特征图，鉴别器必须区分特征图来自哪个网络。鉴别器和特征提取器的总体对抗损失如下
    
![paper4](https://github.com/xuezc/knowledge_distillation/blob/master/images/image14.png)  

&emsp;&emsp;特征提取器G1和G2接受输入x并生成特征图。鉴别器D1获取特征图并产生介于0（假）和1（实）之间的标量。如果特征图来自共同训练的网络（在本例中为G2），则训练输出1；如果特征图来自其所属的网络（在本例中为G1），则训练输出0。D1的目标是通过正确区分两种不同的特征图分布来最小化鉴别器损失项LD1，G1的目标是通过欺骗D1使损失项LG1最小化，从而错误地将G1的特征图确定为实数并得到1。每个训练网络的目标是最小化LGk，以模拟对等网络的特征图分布。通过改变两个网络的角色，这种对抗性方案的工作原理完全相同。
    
&emsp;&emsp;通过结合基于logit的损失和基于对抗特征地图的损失，网络1和2的总体损失如下
    
![paper4](https://github.com/xuezc/knowledge_distillation/blob/master/images/image15.png)  
     
&emsp;&emsp;需要注意的是，基于logit的损失项L_logit和基于特征映射的损失项L_G并不是由同一个优化器优化的。实际上，它们是在同一个小批量中交替优化的。在每一次小批量迭代中，将一幅图像推导出一个模型，然后计算一个logit和一个特征图。然后分别计算这两个损失项，并基于这两个损失对网络进行优化，即用基于logit的损失更新参数一次，然后用基于特征图的损失重新更新参数。其中为每个损失项分别优化的原因是它们分别使用了不同的学习率。对抗性损失需要较小的学习率，因此如果使用相同的优化器和相同的学习速率，网络将不会被优化。
    
&emsp;&emsp;文章还提出了一种同时训练两个以上网络的循环学习方案。它将所需的鉴别器的数目减少到K，其中K是参与的网络的数目。这种循环学习框架不仅比双向学习方法需要更少的计算量，而且与其他多网络在线训练方案相比，可以取得更好的效果。具体的结构图如下图所示
    
![paper4](https://github.com/xuezc/knowledge_distillation/blob/master/images/image16.png)  
    
&emsp;&emsp;在网络中，每个网络以单向循环的方式将其知识传输到下一个对等网络。如果一起训练K个网络，每个网络将其知识提取到下一个网络，除了最后一个网络将其知识转移到第一个网络之外，创建一个循环的知识转移流为1>2,2>3 ,…,(K-1)>K,K>1。使用这种循环学习框架的主要好处是避免使用过多的鉴别器。如果对每一对网络应用提出的对抗损失，它将需要两倍于每一对可能的K网络的数量，这将需要大量的计算。经验表明，对于多个网络，循环训练方案优于其他在线方法的训练方案。
2.	Knowledge Squeezed Adversarial Network Compression(AAAI 2020)：受网络规模差距过大，小网络无法完美模拟大网络的假设启发，文章提出了一种在对抗性训练框架下学习学生网络的知识转移方法，包括有效的中间监督。为了实现功能强大和高度紧凑的中间信息表示，压缩的知识通过任务驱动的注意机制来实现。这样，教师网络的知识转移就可以适应学生网络的规模。结果表明，该方法综合了面向过程学习和面向结果学习的优点。
    
![paper5](https://github.com/xuezc/knowledge_distillation/blob/master/images/image17.png)      
    
&emsp;&emsp;文章提出的KSANC方法的结构图如上图所示，网络主要分为两个部分：主干网络子网络和注意机制子网络。其中注意机制子网络的注意估计器的结构如下图所示
    
![paper5](https://github.com/xuezc/knowledge_distillation/blob/master/images/image18.png)      
    
&emsp;&emsp;其作用是产生压缩知识描述子，进而通过压缩知识描述子来计算中间层的损失。计算压缩知识描述子需要中间层的特征图和全局描述子即softmax的输入向量，具体的计算公式如下
    
![paper5](https://github.com/xuezc/knowledge_distillation/blob/master/images/image19.png)      

其中，W是一个卷积核，用来得到注意力得分M。
    
&emsp;&emsp;对于损失函数，网络的整体损失函数由三部分组成：对抗损失L_adv，主干损失L_b和中间层损失L_is:
    
![paper5](https://github.com/xuezc/knowledge_distillation/blob/master/images/image20.png)  
    
&emsp;&emsp;主干损失通过最小化教师和学生网络主干的logit的L2损失，即
    
![paper5](https://github.com/xuezc/knowledge_distillation/blob/master/images/image21.png)  
    
&emsp;&emsp;对抗损失由三部分组成，在网络实现中使用了GAN网络
    
![paper5](https://github.com/xuezc/knowledge_distillation/blob/master/images/image22.png)   
    
&emsp;&emsp;在式中第一项为由判别器来使学生网络和教师网络的logits输出相近，其公式为
    
![paper5](https://github.com/xuezc/knowledge_distillation/blob/master/images/image23.png)  
    
&emsp;&emsp;第二项引入正则化和类别级监督，进一步改进了判别器，具体使用了三个正则化来增强学生与鉴别器之间的极大极小对策，如下所示，其中w_D为鉴别器的参数
    
![paper5](https://github.com/xuezc/knowledge_distillation/blob/master/images/image24.png)   
    
&emsp;&emsp;第一项只关注在logits分布级别上的匹配，而缺少类别信息，这可能会导致logits和标签之间的不正确关联，第三项进一步修改了鉴别器以同时预测“教师/学生”和类标签，其中l(x)为样本x的标签，C为向量的切片
    
![paper5](https://github.com/xuezc/knowledge_distillation/blob/master/images/image25.png)  
    
&emsp;&emsp;中间层损失通过计算压缩知识描述子的L2损失得到，即
    
![paper5](https://github.com/xuezc/knowledge_distillation/blob/master/images/image26.png)  
 
3.	Knowledge Distillation from Internal Representations(AAAI 2020)：文章指出了当教师规模相当大时，并不能保证教师的知识会转移到学生身上；即使学生与软标签紧密匹配，其内部表现也可能大不相同。这种内部的不匹配可能会破坏最初从教师转移到学生身上的泛化能力。文中除了使用KL散度之外，使用了余弦相似度损失用以转移中间层的暗知识。
    
&emsp;&emsp;具体的，文中的损失函数分为两部分：KL散度和余弦相似度损失。其中KL散度的公式为
    
![paper6](https://github.com/xuezc/knowledge_distillation/blob/master/images/image27.png)  
    
&emsp;&emsp;而余弦相似度损失为
    
![paper6](https://github.com/xuezc/knowledge_distillation/blob/master/images/image28.png)  

其中，h为隐层向量表示。
    
&emsp;&emsp;与其他文章不同的是，文中主要针对BERT模型进行蒸馏操作。对于语言模型，教师网络的不同层次会捕捉不同的语言概念。最近的研究表明，当从网络的底部移动到顶部时，BERT构建的语言属性会变得更加复杂。由于模型建立在底层表示的基础上，除了同时提取所有内部层之外，文章考虑了以自底向上的方式逐步提取与内部表示匹配的知识，因此文章提出了一种渐进内蒸馏和一种堆内蒸馏方法。
    
&emsp;&emsp;在渐进内蒸馏法中，先从下层（接近输入）提取知识，然后逐步向上层移动，直到模型只聚焦类别蒸馏。每一次只优化一个层。在下图中，损失的传递方向为1>2>3>4。
    
![paper6](https://github.com/xuezc/knowledge_distillation/blob/master/images/image29.png)  
    
&emsp;&emsp;在堆内蒸馏方法中，先从较低层提取知识，而不是从一层移到另一层，而是在移到顶层时保留先前层叠加它们所产生的损失。一旦到达顶部，只进行分类，在上图中，损失的传递方式为1>1+2>1+2+3>1+2+3>4。
4.	Structured Knowledge Distillation for Dense Prediction：文章是在作者以前研究的语义分割的基础上，扩展到了将结构信息从大型网络传输到小型网络来进行密集预测任务，文章具体研究了两种结构化的蒸馏方案，一是通过建立静态图来提取成对相似度的成对蒸馏，目标是将一个静态亲和图对齐，该亲和图计算用于从紧凑网络和复杂网络中捕获不同位置之间的短距离和长距离结构信息。另一个是使用对抗训练来提取整体知识的整体蒸馏，目的是将紧凑型网络和复杂网络所产生的输出结构之间的高阶一致性对齐，具体采用了对抗训练，并采用全卷积网络，也就是说，鉴别器同时考虑输入图像和输出结构以产生一个整体嵌入来代表结构的质量。这样使紧凑网络生成与复杂网络相似的嵌入结构，并将结构质量知识提取到判别器的权值中。
    
&emsp;&emsp;文章通过对语义分割、深度估计和目标检测这三个密集预测任务的大量实验，来证明提出方法的有效性。以语义分割作为实例，具体的网络结构图如下图所示
    
![paper7](https://github.com/xuezc/knowledge_distillation/blob/master/images/image30.png)  
    
&emsp;&emsp;由图中的黄色块可以看出，知识蒸馏部分的损失有三部分：逐像素损失(Pixel-wise loss)，逐像素对损失(Pair-wise distillation)和整体损失(Holistic loss)。
    
&emsp;&emsp;对于逐像素损失，文中将分割问题看作是一组独立的像素标记问题，并直接使用知识蒸馏来对齐由紧凑网络产生的每个像素的类概率，由复杂模型产生的类概率作为训练紧致网络的软目标，即为计算logits的KL散度，计算的公式为
    
![paper7](https://github.com/xuezc/knowledge_distillation/blob/master/images/image31.png)  
    
&emsp;&emsp;对于逐像素对损失，文中建立了一个静态亲合图来表示空间对关系，其中，节点代表不同的空间位置，两个节点之间的连接代表相似性，并表示每个节点的连接范围alpha和粒度belta来控制静态关联图的大小。对于每个节点，只根据空间距离和空间局部面片中的聚集的belta个像素来考虑与顶部alpha附近节点的相似性，以表示该节点的特征，如下图所示。
    
![paper7](https://github.com/xuezc/knowledge_distillation/blob/master/images/image32.png)  
    
&emsp;&emsp;对于一个WHC的特征图，有WH个像素，粒度为belta和连接范围为alpha时，静态亲合图包含了WH/belta个节点和WH/belta*alpha个连接。以a为节点之间的相似性，则逐像素对损失的计算公式为
    
![paper7](https://github.com/xuezc/knowledge_distillation/blob/master/images/image33.png)  

其中，两个节点的相似度a通过聚合的特征计算
    
![paper7](https://github.com/xuezc/knowledge_distillation/blob/master/images/image34.png)
    
&emsp;&emsp;对于整体损失，文章将复杂和紧凑网络产生的分割图之间的高阶关系对齐，计算分割图的整体嵌入作为表示。具体实现采用了条件生成对抗学习，将紧致网络视为基于输入RGB图像I的生成器，将预测的分割图Q^s视为假样本。我们期望Q^s与Q^t尽可能相似，Q^t是教师预测的分割图，被视为真实的样本，则整体损失为
    
![paper7](https://github.com/xuezc/knowledge_distillation/blob/master/images/image35.png)

其中，E为期望算子，D是一个具有五个卷积的全卷积神经网络，作为嵌入网络，在GAN中作为判别器，将Q和I一同映射为整体嵌入分值。
    
&emsp;&emsp;最后，总的损失函数为
    
![paper7](https://github.com/xuezc/knowledge_distillation/blob/master/images/image36.png)

其中，l_mc为传统的多类交叉熵损失。

## 三．相关应用
1.	DOMAIN ADAPTATION VIA TEACHER-STUDENT LEARNING FOR END-TO-END SPEECH RECOGNITION：文章指出，在混合语音识别系统中，教师学生结构（T/S）对深度神经网络声学模型的域自适应是有效的。通过两个层次的知识转移，文中将T/S学习扩展到基于attention的端到端（E2E）模型的大规模无监督领域适应：教师标志性的后验概率作为软标签，以及一个最佳预测作为解码器指导。为了进一步提高利用ground-truth标签的T/S学习，文中提出了自适应T/S（AT/S）学习。在AT/s中，学生不必有条件地从教师的软标记后验或一个one-hot ground-truth标签中进行选择，而是通过一对分配给软标记的自适应权重和一个量化每个知识源的置信度的one-hot标签从教师和ground-truth中学习。在解码时，根据软标签和one-hot的函数动态估计置信度得分。文中通过3400小时的并行近距离通话和用于域适应的远场Microsoft Cortana数据，与使用相同数量的远场数据训练的强E2E模型相比，T/S和AT/S的相对字错误率分别提高了6.3%和10.3%。
    
    &emsp;&emsp;对于无监督域自适应，文章通过引入两级知识转移将T/S学习扩展到AED模型：除了从教师的软后验概率外，学生AED还将其解码器条件设置在教师AED解码的one-hot标志性序列上。
    
    &emsp;&emsp;文章进一步提出了一种自适应T/S（AT/S）学习方法，以改进基于ground-truth标签的T/S学习。AT/S利用IT/S和CT/S的优势，根据每个标签上的置信度得分，自适应地为教师的软后验概率和每个解码步骤的one-hot ground-truth标签分配一对权重。将置信度作为软标签和one-hot标签的函数进行动态估计。学生AED从两个标签的自适应线性组合中学习。AT/S继承了IT/S中软标签和one-hot标签的线性插值，并借鉴CT/S对两个知识源合并前的可信度判断。与其他领域自适应的T/S方法相比，更有望获得更好的性能。AT/S作为一种通用的深度学习方法，可广泛应用于任何DNN的领域自适应或模型压缩。
    
    ![paper8](https://github.com/xuezc/knowledge_distillation/blob/master/images/image37.png)
    
    &emsp;&emsp;对于无监督域自适应，网络结构图如上图所示。具体通过计算教师和学生AED的输出分布的KL散度来得到损失，计算公式如下
    
    ![paper8](https://github.com/xuezc/knowledge_distillation/blob/master/images/image38.png)
    
    &emsp;&emsp;对于序列级T/S学习，则将序列级T/S损失函数最小化
    
    ![paper8](https://github.com/xuezc/knowledge_distillation/blob/master/images/image39.png)
    
    &emsp;&emsp;而对于有监督域适应，文章提出了一种利用CT/S和IT/S的自适应师生（AT/S）学习方法，不会为所有解码器步骤分配一对固定的软权重和one-hot权重。其结构图如下所示
    
    ![paper8](https://github.com/xuezc/knowledge_distillation/blob/master/images/image40.png)
    
    &emsp;&emsp;在AT/S中，在每个解码步骤中，分别由w和（1-w）加权的教师软后验和one-hot ground-truth的线性组合被用作学生AED的训练目标。AT/S的损失函数公式如下
    
    ![paper8](https://github.com/xuezc/knowledge_distillation/blob/master/images/image41.png)
    
    &emsp;&emsp;其中，w的计算公式为
    
    ![paper8](https://github.com/xuezc/knowledge_distillation/blob/master/images/image42.png)
    
    &emsp;&emsp;式中的c为置信度分值，d为ground-truth，其计算公式为
    
    ![paper8](https://github.com/xuezc/knowledge_distillation/blob/master/images/image43.png)
    
    &emsp;&emsp;其中的f1和f2都是区间[0：1]上的单调递增函数，在文中使用了
    
    ![paper8](https://github.com/xuezc/knowledge_distillation/blob/master/images/image44.png)

## 四.自我蒸馏
1.  MSD: Multi-Self-Distillation Learning via Multi-classifiers within Deep Neural Networks：随着神经网络的发展，越来越多的深层神经网络被用于图像分类等各种任务中。然而，由于计算开销巨大，这些网络无法应用于移动设备或其他低延迟场景。为了解决这一难题，多出口卷积网络提出允许更快的推断通过早期出口与相应的分类器。这些网络利用复杂的设计来增加早期的退出精度。然而，对多出口网络进行短时训练会损害深度神经网络作为早期出口分类器的性能（准确性），干扰整个特征生成过程。
    
    &emsp;&emsp;文章提出了一种通用的多自蒸馏学习（MSD）训练框架，在同一网络中挖掘不同分类器的知识，提高每个分类器的精度。该方法不仅可以应用于多出口网络，而且还可以用附加的侧分支分类器增强CNNs（例如，RESNET系列）。并可以使用基于采样的分支增强技术将单个出口网络转换为多出口网络，这减少了不同分类器之间的容量差距，提高了MSD的应用效率。实验表明，MSD提高了各种网络的准确率：对于现有的多出口网络（MSDNET），显著提高了每个分类器的准确度，提高了具有内部分类器的高精度单出口网络，同时也提高了最终的准确度。
    
    &emsp;&emsp;对于提出的多自蒸馏学习框架，其中多出口网络中的分类器协作学习并在整个训练过程中相互教授。自互学习框架的一个显著优点是不需要传统的相互学习所需要的其他协作学习模式。（网络中的所有分类器都被训练成学生模型，有效地对下一个最有可能具有不同级别特征的类的集体估计集合起来。具体来说，每个分类器都有三个损失：一个是传统的有监督学习损失，一个是将每个分类器类与其他分类器的类概率相对应的预测模拟损失，一个是将所有分类器的特征映射归纳为适合最深分类器的特征映射的特征模拟损失。最后一个损失考虑了由混合的最深分类器和浅分类器组成的异质队列，使得学习更有效（或多或少）偏向于主动（最深）分类器。
    
    ![paper9](https://github.com/xuezc/knowledge_distillation/blob/master/images/image45.png)
    
    &emsp;&emsp;文章提出方法的网络结构如上图所示，在图中，说明了一个改进的RESNET风格网络，它配备了多个出口。在Resnet样式的网络中，每个图层组包含多个Resnet块，并要调整先前的特征图维度：缩小特征图的宽度和高度尺寸，增加通道尺寸。为了使早期出口的特征映射维数变化模式与骨干网络相似，分别设置第一、第二和第三出口，分别为3, 2和1 ResNet层。这些额外的ResNet层是提出的基于采样的分支扩充架构的一个实例。相对于整个网络，基于采样的分支扩展所增加的计算量可以忽略不计。然而，根据实验结果，这些块带来了极大的精度提高。
    
    &emsp;&emsp;对于具有M类的数据集X，类为Y，网络有n个出口，输出为a^n，则预测概率为
    
    ![paper9](https://github.com/xuezc/knowledge_distillation/blob/master/images/image46.png)
    
    &emsp;&emsp;对于损失函数，分为标签损失、蒸馏损失和特征损失三部分。其中，标签损失来自数据集提供的标签y。对于每个出口，计算类概率和y之间的交叉熵，以这种方式，标签y引导每个出口的适当概率尽可能高，具体的公式为
    
    ![paper9](https://github.com/xuezc/knowledge_distillation/blob/master/images/image47.png)
    
    &emsp;&emsp;对于蒸馏损失，和一般情况一样通过计算教师学生网络输出的KL散度。由于网络有多个出口，对于每个出口，把所有其他N-1个出口作为其教师网络。由于不同的教师提供不同的知识，这样可以使网络的禁锢更高。最后使用平均损失作为每个出口的蒸馏损失
    
    ![paper9](https://github.com/xuezc/knowledge_distillation/blob/master/images/image48.png)
    
     &emsp;&emsp;对于特征损失，通过计算最终FC层之前特征映射之间的L2距离
    
     ![paper9](https://github.com/xuezc/knowledge_distillation/blob/master/images/image49.png)
    
     &emsp;&emsp;最终，网络的整体损失函数为
    
     ![paper9](https://github.com/xuezc/knowledge_distillation/blob/master/images/image50.png)
     
     &emsp;&emsp;其中，alpha和belta为两个超参数，尤其对于belta，采用了余弦退火策略，计算的公式为
    
     ![paper9](https://github.com/xuezc/knowledge_distillation/blob/master/images/image51.png)

## 五．相互学习蒸馏
1. Online Knowledge Distillation with Diverse Peers(AAAI 2020)：蒸馏是一种有效的知识转移技术，它以一个强大的教师模型的预测分布为软目标来训练一个参数化程度较低的学生模型。然而，并非总是有一个经过预先训练的高能力教师。最近提出的在线变体使用多个学生模型的聚合中间预测作为目标来训练每个学生模型。虽然组衍生目标为教师自由蒸馏提供了一个很好的方法，但组成员通过简单的聚合函数快速均匀化，导致早期饱和。

&emsp;&emsp;文章提出了基于不同同伴的在线知识蒸馏（OKDIP），在执行两级蒸馏过程中与多个辅助同伴和一个组长同时训练。在第一级蒸馏中，每个辅助节点具有由基于注意力的机制生成的个体集合权重，以从其他辅助同伴的预测中得到其自身的目标。从不同的目标分布中学习有助于提高基于组蒸馏的有效性。接着执行第二级蒸馏，以将辅助同伴的集合中的知识进一步传递给组长，即用于推理模型。实验结果表明，所提出的框架始可以提供比现有技术更好的性能，而不牺牲训练或推理的复杂性，可以证明所提出的两级蒸馏框架的有效性。

![paper10](https://github.com/xuezc/knowledge_distillation/blob/master/images/image52.png)

&emsp;&emsp;提出方法的网络结构如上图所示，在图中，所有m个学生模型包括m- 1个辅助同伴和一个组长，均使用相同的网络体系结构，都包括了以生成高级特征的特征提取器，接着由分类器生成logit。为了便于表示，所有的学生模型都被索引为1到m，其中1到m -1表示辅助同伴，m对应于组长。第a个学生模型的预测分布用q_a表示，a=1,…,m。

&emsp;&emsp;在第一级蒸馏中，每个辅助同伴a＝1,..,m-1从它自己组的软目标t_a中蒸馏，具体使通过聚合具有不同权重的所有同伴的预测来计算的

![paper10](https://github.com/xuezc/knowledge_distillation/blob/master/images/image53.png)

其中，alpha表示同伴b在a中的参与程度。简单地平等地对待所有的同伴会使经过提炼的模型受到低质量预测的负面影响。因此期望权重能够反映同伴对一个精化模型的相对重要性。为此，通过线性变换将每个同伴模型h_a的特征分别投影到两个子空间中

![paper10](https://github.com/xuezc/knowledge_distillation/blob/master/images/image54.png)

其中W_L和W_E是由所有辅助同伴共享的学习投影矩阵。与自我注意类似，alpha被计算为嵌入高斯距离并进行归一化

![paper10](https://github.com/xuezc/knowledge_distillation/blob/master/images/image55.png)

&emsp;&emsp;那么，所有的辅助同伴的蒸馏损失为

![paper10](https://github.com/xuezc/knowledge_distillation/blob/master/images/image56.png)

&emsp;&emsp;接着，在第二级蒸馏中，将这些辅助同伴的组知识进一步蒸馏到组长，即第m个学生模型，这样通过一级蒸馏的多样性增强，就可以简单地平均所有辅助同伴的预测来计算t_m，则第二级蒸馏的损失函数为

![paper10](https://github.com/xuezc/knowledge_distillation/blob/master/images/image57.png)

&emsp;&emsp;则网络的整体损失函数为

![paper10](https://github.com/xuezc/knowledge_distillation/blob/master/images/image58.png)

其中，L_gt为类概率和标签的交叉熵。

## 六.多教师蒸馏

1. FEED: Feature-level Ensemble for Knowledge Distillation(AAAI20 pre)：知识蒸馏的目的是通过在训练阶段向学生网络提供教师网络的预测，帮助学生网络更好地泛化，从而在师生框架中传递知识。它既可以使用一个高能力的教师，也可以使用多个教师的集合。然而，当人们想要使用基于特征映射的蒸馏方法时，后者并不方便。为了解决这一问题，本文提出了一种通用的、功能强大的知识提取特征集成（FEED）训练算法，其中介绍了一些训练算法，这些算法在特征映射层向学生传递集成知识。在基于特征映射的蒸馏方法中，利用多个非线性变换并行地传递多个教师的知识，帮助学生找到更普遍的解。因此将此方法命名为并行FEED，在CIFAR-100和ImageNet上的实验结果表明，提出的方法在测试时没有引入任何额外的参数或计算，性能有明显的提高。

&emsp;&emsp;假设在特征映射层提取知识和从多个网络集合中提取知识都有各自的优点，我们希望这两种方法能够相互配合。针对这一问题，文章提出了一种训练结构pFEED，用于在特征映射层传递集成知识。N表示教师网络的个数，对每个教师使用不同的非线性变换层，这意味着如果有N个教师，就有N个不同的非线性层。将学生网络的最终特征映射输入到非线性层中，训练其输出以模拟教师网络的最终特征映射。这样，就同时利用了集成模型和基于特征的方法。提出的方法如图2所示，其中NTL是非线性变换层的缩写，每个教师网络分配一个NTL。在训练过程中，所有的教师网络都是固定的，学生网络和NTL网络同时训练。

![paper11](https://github.com/xuezc/knowledge_distillation/blob/master/images/image59.png)

&emsp;&emsp;使用N个不同的教师网络的损失函数如下

![paper11](https://github.com/xuezc/knowledge_distillation/blob/master/images/image60.png)

![paper11](https://github.com/xuezc/knowledge_distillation/blob/master/images/image61.png)

这里，L_CE是交叉熵损失，y是ground-truth标签，s是预测logits，σ（·）表示softmax函数。L_FEED是来自第n个教师网络的损失，x^T_n是来自第n个教师网络的输出特征图，x^S是来自学生网络的输出特征图。NTL_n（·）是第n个非线性转换层，用于使学生网络适应第n个教师网络。每个NTLN（·）由三个核大小为3的卷积层组成，以扩大感受野，使学生能够灵活地融合不同教师网络上的知识。

&emsp;&emsp;BAN法使用交叉熵损失和KD损失，但不软化softmax logits。他们使用一个经过训练的学生网络作为新教师，用来训练一个新学生并递归地这样做。文章利用这种结构上的优势递归地使用同一类型的网络，因为它是一个适合积累和组装知识的模型。通过多次递归地进行知识转移，它还可以集成多个训练序列的知识。文章递归地应用FEED并将这个框架命名为sFEED，其的训练过程如下图所示

![paper11](https://github.com/xuezc/knowledge_distillation/blob/master/images/image62.png)