for noise_mode in     'openset' 'closeset'
do
    for noise_pattern in 'symmetric'
    do
        for noise_rate in 0.45
        do
            for encoder in 'pcnn'
            do
                lr=0.2
                if [ "${encoder}" = "lstm" ];then
                    lr=0.5;
                fi
                python train_bag_cnn_contrastive.py  --log_file ../model_acc/RobustRE/${noise_mode}/${encoder}_${noise_rate}_woSelect_0623.txt  \
                --ckpt RobustRE_${noise_mode}_${encoder}_${noise_rate}_0623  --encoder ${encoder}  --noise_rate ${noise_rate}  --max_epoch 200  --lr ${lr}  \
                --noise_mode ${noise_mode}  --noise_pattern ${noise_pattern} --num_gradual 10 --exponent 0.5  --batch_size 200 
            done
        done
    done
done