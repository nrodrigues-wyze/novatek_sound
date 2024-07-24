setup env
'''
cd nt9856x_sdk_ltd_version_v1.01.007/NT9856x_SDK_Package/nt9856x_linux_sdk_release_uclibc_v1.01.007/
source build/env_setup.sh

Save audio files to sdcard, after the float2fixed function
'''


1. Save ten normalized floating data to bin files
	FILE *fp = fopen("mfcc_clamp_normalized_float.bin", "wb");
	fwrite((void*)sounddata, 4, size_data , fp);
	fclose(fp);
	
2. Use these ten bin files as ref image in aitool, 
   and set [preproc/in/type] = 6, refer to wyze_sounddetect_config_float32.zip
   
3. After run ./closeprefix/bin/compiler.nvtai, go "output\wyze_sounddetect\gentool\siminfo.txt"
   to check input data format, in this case:
    |-+ preproc_sdk
    |-- preproc_hw_en : 0
    |-- rgb2bgr_en : 0
    |-- eltwise_meansub_en : 1
    |-- rotate_en : 0
    |-- permute_en : 0
    |-+ in_size
    |-- dim : 64 101 1 1 1
    |-- len : 6464
    |-+ out_size
    |-- dim : 64 101 1 1 1
    |-- len : 6464
    |-+ in_bit
    |-- bitdepth : 16
    |-- sign_bit_num : 1
    |-- int_bit_num : 8
    |-- frac_bit_num : 7
	
4. according to 3., change the gen_config.txt setting to [preproc/in/type] = 2, [preproc/in/frac_bit_num] = 7
   you can refer to wyze_sounddetect_config_int16.zip
   
5. Run ai_net_get_input_fmt_sample.zip to get input fmt.
   In this case, fmt(0xa1100807)

6. Then saving ten normalized floating data as int16 after converting to int16 by vendor_ai_cpu_util_float2fixed 
   
7. Using these ten int16 files as ref data to generate nvt_model.bin
