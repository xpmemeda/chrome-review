# 如何屏蔽单机

P6n首页 -> 全部应用 -> 微信容灾 -> 屏蔽系统

https://poseidon.woa.com/web/main/dialTest/block/#/block/blockOperation

# 如何停止和重启线上服务

```bash
/usr/bin/python2 /usr/bin/supervisorctl -c /home/qspace/${module}/etc/supervisor_test.conf stop all
```

```bash
/home/qspace/${module}/bin/restart.sh
```

# 如何登陆SG模块查询日志

方案一：登陆SG机器；登录跳板机 -> go mmossrulesz12 -> sggo 业务模块。登录SG模块除了要求登录人是模块的负责人之外，还需要SG人员(stephenw)额外的审批。

方案二：[浏览器页面](https://sg.woa.com/#/https://sg.woa.com/wego/wejobsweb/page?tpl=execution)发送查询请求

# 如何给SG模块发送请求

https://mmbiz.woa.com/poc/#/tester?pn=mmtfcckvproxy&m=AddBusiness


# 如何在开发机上配置patchbuild环境

相关文档：[patchbuild使用相关问题汇总](https://iwiki.woa.com/p/178248830)、[开发机上使用ssh公钥访问git](https://iwiki.woa.com/p/975445323)

通过``git config --global --edit``查看和修改配置。

1. 新建QQMail目录，新建PATCHBUILD_ROOT文件
```bash
mkdir ~/QQMail
echo -n ~/QQMail/PATCHBUILD_ROOT
```

2. 修改git配置
```bash
git config --global url."git@git.woa.com:".insteadOf "http://git.code.oa.com/"
git config --global url."git@git.woa.com:".insteadOf "git@git.code.oa.com:" --add
git config --global url."git@git.woa.com:".insteadOf "http://git.woa.com/" --add
git config --global url."git@git.woa.com:".insteadOf "https://git.woa.com/" --add
```

3. 下载patchbuild仓库，配置PATH
```bash
git clone http://git.woa.com/wxg-mmtest/patchbuild.git ~/local/patchbuild
ln -s ~/local/patchbuild/patchbuild ~/local/patchbuild/pb
export PATH=~/local/patchbuild:$PATH
```

4. 找一个含BUILD文件的仓库，用于生成PATCHBUILD_ENV_CONF
```bash
GIT_LFS_SKIP_SMUDGE=1 git clone git@git.woa.com:wxg-tad/prgateway.git ~/QQMail/prgateway
GIT_LFS_SKIP_SMUDGE=1 git clone git@git.woa.com:wxg-tad/tfccserving.git ~/QQMail/prgateway2/tfccserving
cd ~/QQMail/prgateway/tfccserving/mmtfccbroker && patchbuild env
```
