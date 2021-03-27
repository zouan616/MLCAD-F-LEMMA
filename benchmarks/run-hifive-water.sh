
export CURR_BENCHMARK=splash2-water.nsq
mkdir -p correlation-4-core-$CURR_BENCHMARK;
for i in {1..100};
do echo "running deeper network for $CURR_BENCHMARK ${i}";
./run-sniper -p $CURR_BENCHMARK -i small -n 4 -scontinuous_ipc_power_water:500000 >> correlation-4-core-$CURR_BENCHMARK/result_${i}.txt;
done;
