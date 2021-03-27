export CURR_BENCHMARK=splash2-radix;

for i in {1..100};
do echo "running benchmark $CURR_BENCHMARK ${i}";
./run-sniper -p $CURR_BENCHMARK -n 4 -i small -scategorical_without_pacman:500000 >> radix-more-layers/result_${i}.txt;
done;
