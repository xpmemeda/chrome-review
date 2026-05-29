# Bernard

## 怎么从 Bernard 实例上面下载文件

自己启动一个 python http 服务

```bash
python3 -m http.server --bind :: 9320
```

这里 ``--bind :: `` 的意思是要绑定所有网络地址，包括 ipv6，不然就只会监听 ipv4 端口。

下载命令，举个例子。

```bash
wget "http://[2605:340:cd51:5800:66b1:75a1:450e:8d2b]:8000/config.yaml"
```

ipv6 的地址要用中括号包起来，不然冒号被识别为端口，命令解析失败。

## 怎么上传文件到 Bernard 服务实例

在服务实例上起一个服务

```bash
python -m uploadserver --bind :: 8000
```

从浏览器打开 `http://[ipv6]:8000`，会有拖拽上传的窗口。

# Merlin

## Merlin 开发机下载内网代码过不了身份认证

比如

```bash
git clone git@code.byted.org:ocean/service_shell.git
```

Merlin 开发机默认生成了一份 `~/.ssh/config`，其中给 `*.byted.org` 配的私钥路径是 `~/.ssh/gitlab_rsa`。所以要改一下 `~/.ssh/config` 文件，把 `*.byted.org` 指定的私钥路径改回 `~/.ssh/id_rsa.pub`。

## Merlin 开发机下载 github 代码非常慢

设置代理

```bash
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy="localhost,.byted.org,byted.org,.bytedance.net,bytedance.net,.byteintl.net,.tiktok-row.net,.tiktok-row.org,127.0.0.1,127.0.0.0/8,169.254.0.0/16,100.64.0.0/10,172.16.0.0/12,192.168.0.0/16,10.0.0.0/8,::1,fe80::/10,fd00::/8"
```

## Merlin 怎么看当前有哪些 GPU worker 以及如何登陆

看 GPU worker

```bash
mlx worker list
```

登陆 GPU worker

```bash
mlx worker login ${worker_id}
```

## Merlin 开发机配置 Vibe coding CLI (TRAE CLI)

参考飞书文档：[TRAE CLI (Coco CLI)](https://bytedance.larkoffice.com/wiki/STuTw58XjifuU5kK2AmcFk0vnyf)
