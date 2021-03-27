
export CURR_BENCHMARK=splash2-fft;
export DIR_NAME=local_swift_4-$CURR_BENCHMARK;
mkdir -p $DIR_NAME;
for i in {1..100};
do echo "running deeper network for $CURR_BENCHMARK ${i}";
./run-sniper -p $CURR_BENCHMARK -i small -n 4 -scontinuous_ipc_power:500000 >> $DIR_NAME/result_${i}.txt;
done;
