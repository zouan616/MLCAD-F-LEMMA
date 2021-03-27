
export CURR_BENCHMARK=local-pi;
mkdir -p test-${CURR_BENCHMARK};
./run-sniper -p $CURR_BENCHMARK -i small -n 4 -ssample_state_space:500000 >> test-${CURR_BENCHMARK}/result_0.txt;
for i in {1..100};
do echo "running deeper network for $CURR_BENCHMARK ${i}";
./run-sniper -p $CURR_BENCHMARK -i small -n 4 -scontinuous_ipc_power_local:500000 >> test-${CURR_BENCHMARK}/result_${i}.txt;
done;
