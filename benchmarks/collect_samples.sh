
export CURR_BENCHMARK=parsec-bodytrack;
mkdir -p deeper-network-$CURR_BENCHMARK;
for i in {1..100};
do echo "collecting data for $CURR_BENCHMARK ${i}";
./run-sniper -p $CURR_BENCHMARK -i simsmall -n 4 -scategorical_without_pacman.py:500000 >> deeper-network-$CURR_BENCHMARK/result_${i}.txt;
done;
