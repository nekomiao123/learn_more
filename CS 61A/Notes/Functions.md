# Functions

[TOC]

## Expressions

An expression describes a computation and evaluates to a value

All expressions can use function call notation

### Anatomy of a Call Expression

![image-20210322151706524](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210322151706524.png)

### Evaluating Nested Expressions

![image-20210322151459872](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210322151459872.png)

## Names, Assignment, and User-Defined Functions

![image-20210322151804676](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210322151804676.png)

### A Question

![image-20210322151853818](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210322151853818.png)

Functional Programming(函数式编程,函数式编程的一个特点就是，允许把函数本身作为参数传入另一个函数，还允许返回一个函数!)

[A link about the functional programming in Chinese](https://www.liaoxuefeng.com/wiki/1016959663602400/1017328525009056)

This is a good example of functional programming. The result is 3.

I ran the code in my own computer and I got the result below.

![image-20210322152204165](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210322152204165.png)

## Environment Diagrams

Environment diagrams visualize the interpreter’s process.

[a website can do this](http://pythontutor.com/composingprograms.html#mode=edit)

![image-20210322155236175](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210322155236175.png)

## Defining Functions

Assignment is a simple means of abstraction: binds names to values

Function definition is a more powerful means of abstraction: binds names to expressions

```python
# Function signature indicates how many arguments a function takes
def <name>(<formal parameters>):
    return <return expression>
# Function body defines the computation performed when the function is applied
```

def 执行的步骤

- Create a function with signature <name>**(**<formal parameters>)
- Set the body of that function to be everything indented after the first line. 设置函数的主体（第一行后的所有内容），但并未执行函数主体
- Bind <name> to that function in the current frame. 把 `<name>` 和当前 frame 的 function 绑定

### Calling User-Defined Functions

**Procedure for calling/applying user-defined functions (version 1):**

1. Add a local frame, forming a new environment
2. Bind the function's formal parameters to its arguments in that frame 
3. Execute the body of the function in that new environment

![image-20210322163559054](https://cdn.jsdelivr.net/gh/nekomiao123/pic/img/image-20210322163559054.png)

A function’s signature has all the information needed to create a local frame

### Looking Up Names In Environments

Every expression is evaluated in the context of an environment.

So far, the current environment is either:
 • Theglobalframealone,or
 • Alocalframe,followedbytheglobalframe.

The most important

> An environment is a sequence of frames.
>
> A name evaluates to the value bound to that name in the earliest frame of the current environment in which that name is found.

第二句的大概意思应该是 绑定到这个名称上的值 就是在当前环境下找到这个名字的最早帧



E.g., to look up some name in the body of the square function:

- Look for that name in the local frame.

- If not found, look for it in the global frame. (Built-in names like “max” are in the global frame too,

  but we don’t draw them in environment diagrams.)

这个例子要好理解得多, 看着很像全局变量和局部变量，但是不是 他们比这些还要底层

## Q&A

What is a frame?

It's like a memory where the computer or the interpreter is keeping track of what names mean.

The same name in different frames can mean different things.

What is a environment?

Environment is all the names that you can refer to. It is a sequence of frames.

When the local frame would be called?

The local frame is created when you call the function