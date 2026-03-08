# ip addr 显示的字段网卡类别

|Interface Example  |Type/Full Name     |Description                                                                                                            |
|-                  |-                  |-                                                                                                                      |
|lo                 |Loopback           |Loopback interface for local inter-process communication (127.0.0.1)                                                   |
|eth0, eth1         |Ethernet           |Traditional physical Ethernet network interface                                                                        |
|bond0, bond1       |Bonding/Teaming    |Network interface aggregation for redundancy or load balancing                                                         |
|docker0            |Docker Bridge      |Default virtual bridge created by Docker for container networking                                                      |
|ib0, ib1           |InfiniBand         |High-performance network interface for InfiniBand (HCA); supports RDMA and can be used for IPoIB (IP over InfiniBand)  |
|usb0               |USB Network        |USB network interface, used for USB network adapters or USB tethering                                                  |
|team0              |Teaming            |Similar to bonding, for network interface aggregation                                                                  |


# 如何查看 Linux 系统机器上的 IB 网络设备

```bash
ls /sys/class/infiniband
```

每个设备都可以对应到一个 PCIe 地址。

```bash
readlink /sys/class/infiniband/mlx5_bond_1/device
```

通过 lspci 来查看这块物理设备（也有可能是虚拟的），常见的厂家有 Mellanox。

```bash
lspci | grep 2f:00
```
