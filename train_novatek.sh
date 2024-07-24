#sleep 30m

DATA_CONFIG_FILE="data/wyze_sound_new.data"
MODEL_CONFIG_FILE="cfg/sounddet_16x_8cls_stride_32bit_maxpool_v7_256.cfg"
SAVE_DIR="weights/model_novatek_32bit_may_1_input_255_new_clsweights_2_int16_float/"
RESULTS_FILE="32bit_results_finalt31.txt"

#prev model pretrained jly 13 2024 weights/model_novatek_32bit_16k_group/backup204000.pt
#model_novatek_32bit_16k_group/backup204000.pt

# TEST_DATA_CSV="/data/data/test_set_1103_wyze_clip_00.csv"
# <<comment
python3 train.py \
	--balanced \
	--batch-size 256 \
	--accumulate 2 \
	--epochs 15 \
	--save-dir ${SAVE_DIR} \
	--data ${DATA_CONFIG_FILE} \
	--cfg ${MODEL_CONFIG_FILE} \
	--save_iter_interval 1500 \
	--weights "" \
	--device 0
