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
        local cli_dir="${VSCODE_GIT_ASKPASS_NODE:0:-5}/bin/remote-cli"
        if [[ -x "${cli_dir}/code" ]]; then
            "${cli_dir}/code" "$@"
        elif [[ -x "${cli_dir}/trae-cn" ]]; then
            "${cli_dir}/trae-cn" "$@"
        else
            echo "command not found: code or trae-cn in ${cli_dir}"
        fi
    else
        echo "command not found: code"
    fi
}
alias code-r="code ${HOME}/workspace/chrome-review"
alias r="${HOME}/workspace/chrome-review"
alias gs="git status"
