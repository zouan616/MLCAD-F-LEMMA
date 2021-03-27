
export CURR_BENCHMARK=splash2-fft;
export DIR_NAME=profit-core-4-$CURR_BENCHMARK;
rm -rf $DIR_NAME;
mkdir -p $DIR_NAME;
for i in {1..100};
do echo "running deeper network for $CURR_BENCHMARK ${i}";
./run-sniper -p $CURR_BENCHMARK -i small -n 4 -sprofit:500000 >> $DIR_NAME/result_${i}.txt;
done;
