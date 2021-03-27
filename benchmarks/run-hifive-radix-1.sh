
export CURR_BENCHMARK=splash2-radix;
mkdir -p swift-overhead-reduced-PF-4-core-${CURR_BENCHMARK}-1;
for i in {1..100};
do echo "running deeper network for $CURR_BENCHMARK ${i}";
./run-sniper -p $CURR_BENCHMARK -i small -n 4 -scontinuous_ipc_power_radix_1:500000 >> swift-overhead-reduced-PF-4-core-${CURR_BENCHMARK}-1/result_${i}.txt;
done;
