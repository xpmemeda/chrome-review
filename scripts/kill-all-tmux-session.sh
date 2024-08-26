#!/bin/bash

sessions=$(tmux list-sessions -F "#{session_name}")

for session in $sessions; do
    echo "Closing tmux session: $session"
    tmux kill-session -t "$session"
done

echo "All tmux sessions have been closed."
