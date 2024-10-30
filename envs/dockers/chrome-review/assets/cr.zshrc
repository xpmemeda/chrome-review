export ZSH="$HOME/.oh-my-zsh"
ZSH_THEME="ys"
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
alias gs="git status"

export TMPDIR=${HOME}/.tmp
export GTEST_HOME=${HOME}/local/googletest
