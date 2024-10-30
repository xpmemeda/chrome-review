export RUSTUP_HOME=$HOME/local/rustup
export CARGO_HOME=$HOME/local/cargo

if [ ! -d $CARGO_HOME ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    rustup install stable && rustup default stable
fi
