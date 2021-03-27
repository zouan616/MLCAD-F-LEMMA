
export CURR_BENCHMARK=splash2-fft;
rm -rf swift_overhead20-core-$CURR_BENCHMARK;
mkdir -p swift-overhead20-core-$CURR_BENCHMARK;
for i in {1..100};
do echo "running deeper network for $CURR_BENCHMARK ${i}";
./run-sniper -p $CURR_BENCHMARK -i small -n 4 -scontinuous_ipc_power_20:500000 >> swift-overhead20-core-$CURR_BENCHMARK/result_${i}.txt;
done;
