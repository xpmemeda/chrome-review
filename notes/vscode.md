# Mac系统上使用Vim插件不能长按HJKL来控制光标连续移动

使用终端关闭Mac的“Press and Hold”功能（以下命令仅对VSCode生效）：

```bash
defaults write com.microsoft.VSCode ApplePressAndHoldEnabled -bool false
```

也可以全局设置：

```bash
defaults write -g ApplePressAndHoldEnabled -bool false
```

# Mac系统上使用Vim插件长按HJKL光标移动速度太慢

1. 打开系统偏好设置（点击屏幕左上角的苹果图标，选择“系统偏好设置”）。
2. 选择“键盘”选项。
3. 将“按键重复”滑块调到“快”，将重复前延迟”滑块调到“短”。
