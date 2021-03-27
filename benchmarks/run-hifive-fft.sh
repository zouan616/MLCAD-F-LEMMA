
export CURR_BENCHMARK=splash2-fft;
mkdir -p swift-overhead-reduced-PF-4-core-$CURR_BENCHMARK;
for i in {1..100};
do echo "running deeper network for $CURR_BENCHMARK ${i}";
./run-sniper -p $CURR_BENCHMARK -i small -n 4 -scontinuous_ipc_power:500000 >> swift-overhead-reduced-PF-4-core-$CURR_BENCHMARK/result_${i}.txt;
done;
