# 常见CPU的规格性能对比

**phoronix**

- 安装方式：sudo yum install phoronix-test-suite
- 版本：v10.8.4
- 命令：phoronix-test-suite benchmark pybench
- 指标名称：Total For Average Test Times
- 指标说明：Python解释器的综合执行耗时，越小越好

**sysbench**

- 安装方式：sudo yum install sysbench
- 版本：1.0.20
- 命令：sysbench --threads=1 cpu run
- 指标名称：events per second
- 指标说明：单位时间内查找素数的个数，越大越好

**7zip**

- 安装方式：``sudo yum install p7zip``
- 版本：16.02
- 命令：7za b -mmt1
- 指标名称：Tot MIPS (Higher is Better)
- 指标说明：单位时间内编解码指令的执行数量，越大越好

|CPU                |Vendor |Core(s)|CPU(s) |CPU GHz|L1d    |L1i    |L2     |L3     |phoronix   |sysbench   |7zip   |
|-                  |-      |-      |-      |-      |-      |-      |-      |-      |-          |-          |-      |
|Xeon Platinum 8255C|Intel  |24     |48     |2.5    |32K    |32K    |4096K  |36608K |1400       |1049.30    |3486   |
|EYPC 7K62          |AMD    |41     |82     |2.6    |32K    |32K    |4096K  |16384K |1629       |1647.22    |3669   |
|EPYC 7K83          |AMD    |28     |56     |2.55   |32K    |32K    |512K   |32768K |1184       |2682.14    |4915   |
|EPYC 9K84          |AMD    |96     |192    |2.6    |32K    |32K    |1024K  |32768K |982        |4221.02    |5457   |
|EPYC 9754          |AMD    |16     |32     |2.25   |32K    |32K    |1024K  |16384K |1159       |           |       |
