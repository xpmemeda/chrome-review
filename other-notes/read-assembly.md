### 通过看汇编代码（Assembly）查问题

**GDB指令**

查看当前指令（``x``查看指定内存地址的内容，[详细文档](https://visualgdb.com/gdbreference/commands/x)）：

```
x/10i $pc
```

查看当前函数的汇编代码（即使release编译也有函数签名信息）：
```
disassemble
```

查看常见寄存器的值：

```
p $rsp      栈顶
p $eflags   控制位（ZL,CL...）
p $rax      查看通用寄存器rax的值
```

**汇编指令文档**

[x86 and amd64 instruction reference](https://www.felixcloutier.com/x86/index.html)

**GDB Instruction Operand**

一个16进制表示的常量：
```
0x200
```

在rsp地址的基础上偏移0x4单位，通常指向某个栈内变量的地址：
```
0x4(%rsp)
```

**StackOverflow**：程序运行时Segfault，gdb调试时弹出Cannot access memory at address（访问不可读内存地址）

发生过程：
1. 修改rsp的值，栈溢出发生时，rsp可以正常被修改，但是其值已经超出了栈的范围。
2. 通过rsp来访问栈内的变量，此时可能出现内存越界，程序退出。

解决办法：在挂的地方打印rsp，和调用前的rsp对比，看看是不是差值过大。确定是栈溢出后可以把栈调大。
