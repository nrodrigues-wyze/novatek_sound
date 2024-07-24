#weights/model_oct_4bit_try3/backup280000.pt
#data/test_set_sept.csvweights/model_novatek_32bit_16k_group/backup204000.pt
#weights/model_novatek_32bit_16k_group_relu_input//backup1500.pt



#python3 val.py \
#	--cfg 'cfg/sounddet_16x_8cls_stride_32bit_maxpool_v7_256.cfg' \
#	--model_path='weights/model_novatek_32bit_may_1_old_data_input/backup1500.pt' \
#	--data_path='test_samples.csv' \
#	--device=0 \
#	--batch_size 256 \
#	--results-file "final_results.txt"


#sleep 25m
#/data/data/test_set_1103_wyze_clip_00.csv
declare -a model_paths=('12000' '15000' '45000' '85500')

for i in "${model_paths[@]}"
do
	python3 val.py \
	--cfg 'cfg/sounddet_16x_8cls_stride_32bit_maxpool_v7_256.cfg' \
	--model_path="weights/model_novatek_32bit_may_1_input_255_new/backup$i.pt" \
	--data_path='/data/sounddet_workingwell_novatek_relu/test_set_1103_wyze_clip_00.csv' \
	--device=0 \
	--batch_size 256 \
	--results-file "32bit_results.txt"
done
