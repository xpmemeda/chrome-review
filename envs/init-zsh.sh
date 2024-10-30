yes | sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
git clone https://github.com/zsh-users/zsh-autosuggestions ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting

sudo usermod $(whoami) -s $(which zsh)

SCRIPT_DIR=$(realpath $(dirname "$0"))
if [ "$SCRIPT_DIR" != "$HOME" ]; then
    rm -rf "$HOME"/.zshrc && ln -s "$SCRIPT_DIR"/.zshrc "$HOME"/.zshrc
fi
