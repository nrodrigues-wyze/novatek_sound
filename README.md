Train the model using 
```
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

```


Download data from : wyze-ai-team-individual-work-data/neil/all_train_20240503_real16k.h5