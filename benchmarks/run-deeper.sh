
export CURR_BENCHMARK=splash2-fft;
mkdir -p deeper-16core-$CURR_BENCHMARK;
for i in {1..100};
do echo "running deeper network for $CURR_BENCHMARK ${i}";
./run-sniper -p $CURR_BENCHMARK -i small -n 16 -scategorical_without_pacman.py:500000 >> deeper-16core-$CURR_BENCHMARK/result_${i}.txt;
done;
