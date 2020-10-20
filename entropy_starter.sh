#!/bin/bash

starter="Relative_Entropy"
tmux has-session -t $starter 2>/dev/null
if [ $? != 0 ]; then
    tmux new-session -d -s Relative_Entropy
    tmux send -t Relative_Entropy "bash" ENTER
    tmux send -t Relative_Entropy "cd /cluster/home/besuter/RelativeEntropy" ENTER
    tmux send -t Relative_Entropy "source RelEnt/bin/activate" ENTER
    tmux send -t Relative_Entropy "cd Relative_Entropy/Code" ENTER
fi
Relative_Entropy=$(pgrep -af "python Relative_Entropy.py")
if [ -z "$Relative_Entropy" ]; then
    tmux send -t Relative_Entropy "python Relative_Entropy.py" ENTER
fi