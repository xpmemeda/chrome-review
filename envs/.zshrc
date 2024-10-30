export ZSH="$HOME/.oh-my-zsh"
if [[ $(hostname) == "I5-4070S" ]]; then
    ZSH_THEME="robbyrussell"
else
    ZSH_THEME="ys"
fi
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
source $ZSH/oh-my-zsh.sh

function rkill() { ps -u | grep $1 | awk '{print $2}' | xargs kill -9; }
function rcpu() { top -bn 1 | grep $1 | awk '{sum+=$9;}END{print sum"%"}'; }
function rwatch-cpu() { watch -n 1 "top -bn 1 | grep $1 | awk '{sum+=\$9;}END{print sum\"%\"}'"; }
function rcpus() { top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'; }
function rwatch-cpus() { watch -n 1 "top -bn1 | grep \"Cpu(s)\" | sed \"s/.*, *\([0-9.]*\)%* id.*/\1/\" | awk '{print 100 - \$1\"%\"}'"; }
function cmake() {
    if [[ "$1" == "-b" ]]; then
        command cmake "--build" "${@:2}"
    elif [[ "$1" == "-i" ]]; then
        command cmake "--install" "${@:2}"
    else
        command cmake "$@"
    fi
}
function code() {
    if [[ "$VSCODE_GIT_ASKPASS_NODE" != "" ]] ; then
        ${VSCODE_GIT_ASKPASS_NODE:0:-5}/bin/remote-cli/code "$@"
    else
        echo "command not found: code"
    fi
}
alias code-wnr="code ${HOME}/workspace/wnr"
alias wnr="${HOME}/workspace/wnr"
alias code-tfcc="code ${HOME}/workspace/tfcc"
alias tfcc="${HOME}/workspace/tfcc"
alias code-r="code ${HOME}/workspace/chrome-review"
alias r="${HOME}/workspace/chrome-review"
alias gs="git status"

export TMPDIR=${HOME}/.tmp
export GTEST_HOME=${HOME}/local/googletest

SCRIPT_DIR=$(dirname $(realpath ${HOME}/.zshrc))
if [ "$SCRIPT_DIR" != "$HOME" ]; then
    source $SCRIPT_DIR/init-tx.sh
fi
