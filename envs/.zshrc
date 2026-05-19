export ZSH="$HOME/.oh-my-zsh"

if [[ $(hostname) == "I5-4070S" ]]; then
    ZSH_THEME="robbyrussell"
else
    ZSH_THEME="ys"
fi
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
source $ZSH/oh-my-zsh.sh


SCRIPT_DIR=$(dirname $(realpath ${HOME}/.zshrc))

# System.
if [ "$SCRIPT_DIR" != "$HOME" ]; then
    source $SCRIPT_DIR/init-sys-env.sh
fi

# Alias.
if [ "$SCRIPT_DIR" != "$HOME" ]; then
    source $SCRIPT_DIR/init-alias.sh
fi

# xlib.
if [ "$SCRIPT_DIR" != "$HOME" ]; then
    source $SCRIPT_DIR/init-xlib-env.sh
fi

# Tencent.
if [ "$SCRIPT_DIR" != "$HOME" ]; then
    source $SCRIPT_DIR/init-tencent-env.sh
fi

# Bytedance.
if [ "$SCRIPT_DIR" != "$HOME" ]; then
    source $SCRIPT_DIR/init-bytedance-env.sh
fi
