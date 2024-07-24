python3 generate_pr.py \
	--cfg 'cfg/sounddet_16x_8cls_stride_4bit_maxpool_v7.cfg' \
	--model_path='/data/sounddet_workingwell_6classes/weights/sounddet_model_maxpool_4bit_8cls_sept/backup160000.pt' \
	--data_path='merged_test_set.csv' \
	--device=0 \
	--batch_size 32 \
	--results-file "4bit_results_t40.txt"

