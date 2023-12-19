for ((i=225; i<=400; i+=25)); do
    echo $i
    ./run_experiments.sh 9 Fixed $i
    ./run_experiments.sh 9 Random $i
    ./run_experiments.sh 9 DQN $i
    ./run_experiments.sh 9 Free $i
done
