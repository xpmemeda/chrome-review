alias gs="git status"
alias python="python3"

_chrome_review_dir="$(cd "$(dirname "${(%):-%N}")/../.." && pwd)"
function r() {
    cd "$_chrome_review_dir"
}
