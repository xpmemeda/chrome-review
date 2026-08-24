export ZSH="$HOME/.oh-my-zsh"

if [[ $(hostname) == "I5-4070S" ]]; then
    ZSH_THEME="robbyrussell"
else
    ZSH_THEME="ys"
fi
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
source $ZSH/oh-my-zsh.sh

SCRIPT_DIR=$(dirname $(realpath ${HOME}/.zshrc))

if [[ "$(uname -s)" == "Darwin" ]]; then
    if [ "$SCRIPT_DIR" != "$HOME" ]; then
        source $SCRIPT_DIR/Darwin/init-alias.sh
        source $SCRIPT_DIR/init-codex.sh
        source $SCRIPT_DIR/Darwin/init-vscode.sh
    fi
    return
fi

if [[ "$(uname -s)" == "Linux" ]]; then
    if [ "$SCRIPT_DIR" != "$HOME" ]; then
        source $SCRIPT_DIR/Linux/init-sys-env.sh
        source $SCRIPT_DIR/Linux/init-alias.sh
        source $SCRIPT_DIR/Linux/init-xlib-env.sh
        source $SCRIPT_DIR/Linux/init-tencent-env.sh
        source $SCRIPT_DIR/Linux/init-byted-env.sh
        source $SCRIPT_DIR/init-codex.sh
        source $SCRIPT_DIR/Linux/init-trae.sh
        source $SCRIPT_DIR/Linux/init-vscode.sh
    fi
    return
fi
