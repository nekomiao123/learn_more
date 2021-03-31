# learn more （ML and DL）

## basic

about math

- [convex optimization](https://www.edx.org/course/convex-optimization)
- [linear algebra by 3Blue1Brown](https://www.bilibili.com/video/av6731067?p=1&share_medium=android&share_source=copy_link&bbid=PgppX2ZRaAtpWz9bJ1sninfoc&ts=1552297488727)
- [linear algebra by Strang](https://ocw.mit.edu/resources/res-18-010-a-2020-vision-of-linear-algebra-spring-2020/videos/a-new-way-to-start-linear-algebra/)

about deep learning

- [Deep Learning Specialization](https://www.deeplearning.ai/program/deep-learning-specialization/)

## advanced

### course

- [李宏毅ML2021](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html)
- [李宏毅ML2020 内容非常多](https://speech.ee.ntu.edu.tw/~hylee/ml/2020-spring.html)
- [Berkeley full stack deep learning 2021](https://fullstackdeeplearning.com)
- [CS294-158-SP20 Deep Unsupervised Learning -- UC Berkeley, Spring 2020](https://sites.google.com/view/berkeley-cs294-158-sp20/home)
- [CS182 Designing, Visualizing and Understanding Deep Neural Networks](https://cs182sp21.github.io/)
- [Analyses of Deep Learning (STATS 385)](https://stats385.github.io/)

### books

- [Dive into Deep Learning](https://d2l.ai/index.html) and [配套直播课](https://courses.d2l.ai/zh-v2/)

## computer vision

### course

- [Deep Learning for Computer Vision 2019](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/)

- [Deep Learning for Computer Vision Fall 2020](https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/schedule.html)

### books

- [Computer Vision: Algorithms and Applications, 2nd ed.](https://szeliski.org/Book)

# learn more (Others)

## basic SICP

- [CS 61A 2020 Fall](https://inst.eecs.berkeley.edu/~cs61a/fa20/)
- [CS 61A 2021 spring](https://cs61a.org)

## data Structure

- [CS 61B](https://sp21.datastructur.es/)

## 关于Berkeley的学习顺序

[Berkeley EECS的大部分课程](https://inst.eecs.berkeley.edu//classes-eecs.html)

![course-map-2019-da79ecbe2fc25e6b6349b8931364890032b8d51aacaccea65151ae98316f0587](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/course-map-2019-da79ecbe2fc25e6b6349b8931364890032b8d51aacaccea65151ae98316f0587.png)



参考 https://www.zhihu.com/question/23372616

推荐学习顺序：Math 54 + EE 16A（仅线性代数部分，可以跳过电路）-> Math 110 + EE 16B + CS 70 -> CS 189 -> EECS 126 + EECS 127 -> CS 182

> From my perspective as someone who teaches 189, the following makes the most sense 
> 1) If you have taken 16AB and 110, and have strong vector calculus, the single most important course to prepare for 189 is 126. This continues to hold as long as you have taken 16AB and do not have have some weakness in vector calculus or linear algebra. Being able to reason probabilistically and have intuition about it is very important. 70 is not really enough. 
> 2) If you have weakness in linear algebra or vector calculus or have not taken 16AB, then 127 is the most important course to prepare for 189. You will experience significant difficulties without that understanding. 
> 3) If you have taken 16AB and 127, for many students, taking 110 could become largely redundant. A student in that position should take the 110 textbook and read it on their own. See if you need more. 
> Of course, hopefully everyone understands that both 126 and 127 are *more* fundamental courses to take than 189 if you are interested in machine learning. And if you have to pick only one of the three courses: 126, 127, and 189; then 127 is clearly the most important one to take for students regardless of their area. There are lots of machine learning applications in 127. 
> In light of this, the most reasonable course of action is for interested students to take both 126 and 127 before 189. And if because of time pressure to graduate they want to take one concurrently with 189, that course is likely 189. (Although it is better understood as taking 189 concurrently with 127.) 
> Courses are far more rewarding if you can really understand what is going on. None of these courses are gratuitously difficult or anything like that, but truly engaging with the material does require hard work and building up your understanding. 
> I understand that the fashionable name and interesting subject matter of 189 is appealing, but when you're seven weeks into the course, trust me, no amount of fashionability is going to help you out or even make you feel any better. Understanding probability and optimization well will.

- [CS 189 Introduction to Machine Learning](https://people.eecs.berkeley.edu/~jrs/189/)

线性代数：Math 54 + EE 16A -> Math 110 + EE 16B -> CS 189

Math 54和110就是初级和高级线性代数，注重理论和证明，EE 16A/B是电子工程的基础课，课程会要求你大量应用线性代数的性质去解决电路/系统问题。

概率：CS 70 (+ Stat 134/135) -> CS 189

CS 70的后半部分包括了一些概率，有教授和同学认为这个知识不太够，所以可以通过Stat 134/135来训练一下，但是统计的课程更加注重理论和证明，所以还是会有一些差距。

EECS 126：概率的应用，听说会讨论Google初期的Page Rank。

EECS 127：优化模型，对于掌握线性代数的应用很有帮助。

文中教授建议先上这两门课再进入CS 189，因为这两门课更加基础而纯粹，没有将线性代数和概率结合在一起；但是身边的同学反应这个操作难度有点大，还是建议先上189再进入126/127进一步加强机器学习的能力。

