# Analysis of For Loops 

![image-20210426164744004](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210426164744004.png)

**Geometric argument is more useful**



# The definition of Big-Theta

![image-20210426162325986](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210426162325986.png)

# The Definition of Big O

![image-20210426162736177](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210426162736177.png)

The difference between them:

Whereas Big Theta can informally be thought of as something like “equals”, Big O can be thought of as “less than or equal”.

![image-20210426162901212](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210426162901212.png)

![image-20210426163156849](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210426163156849.png)

# Summary

Given a code snippet, we can express its runtime as a function R(N), where N is some property of the input of the function (often the size of the input).
Rather than finding R(N) exactly, we instead usually only care about the order of growth of R(N).

One approach (not universal):

- Choose a representative operation, and let C(N) be the count of how many times that operation occurs as a function of N.
- Determine order of growth f(N) for C(N), i.e. C(N) ∈ Θ(f(N)) 
  - Often (but not always) we consider the worst case count.
- If operation takes constant time, then R(N) ∈ Θ(f(N))
- Can use O as an alternative for Θ. O is used for upper bounds.