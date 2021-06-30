# The Shell

. means current directory

.. means parent directory

cd ~ means to home directory

mkdir my\ python (we need add \ if we want to create a file which has space )

ctrl + l == clear



## stream and pip

In the shell, programs have two primary “streams” associated with them: their input stream and their output stream.

The simplest form of redirection is `< file` and `> file`

These let you rewire the input and output streams of a program to a file respectively:

```bash
missing:~$ echo hello > hello.txt
missing:~$ cat hello.txt
hello
missing:~$ cat < hello.txt
hello
missing:~$ cat < hello.txt > hello2.txt
missing:~$ cat hello2.txt
hello
```

You can also use `>>` to append to a file.

```bash
missing@ubuntu:~/test$ echo hello >> hello.txt
missing@ubuntu:~/test$ cat hello.txt
hello
hello
```

The `|` operator lets you “chain” programs such that the output of one is the input of another

```bash
missing:~$ ls -l / | tail -n1
drwxr-xr-x 1 root  root  4096 Jun 20  2019 var
missing:~$ curl --head --silent google.com | grep --ignore-case content-length | cut --delimiter=' ' -f2
219
```

