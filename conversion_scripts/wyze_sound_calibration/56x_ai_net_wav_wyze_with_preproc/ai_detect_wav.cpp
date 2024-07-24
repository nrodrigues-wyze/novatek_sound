/**
	@brief Source file of vendor ai net sample code.

	@file ai_net_with_buf.c

	@ingroup ai_net_sample

	@note Nothing.

	Copyright Novatek Microelectronics Corp. 2020.  All rights reserved.
*/

/*-----------------------------------------------------------------------------*/
/* Including Files                                                             */
/*-----------------------------------------------------------------------------*/
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
extern "C"
{
    #include "hdal.h"
	#include "vendor_ai.h"
	#include "hd_type.h"
	#include "hd_debug.h"
}

#include <arm_neon.h>
#include <sys/time.h>
#include <dirent.h>
#include <time.h>

#include "Handlewave.h"


// platform dependent
#if defined(__LINUX)
#include <pthread.h>			//for pthread API
#define MAIN(argc, argv) 		int main(int argc, char** argv)
#define GETCHAR()				getchar()
#else
#include <FreeRTOS_POSIX.h>	
#include <FreeRTOS_POSIX/pthread.h> //for pthread API
#include <kwrap/util.h>		//for sleep API
#define sleep(x)    			vos_util_delay_ms(1000*(x))
#define msleep(x)    			vos_util_delay_ms(x)
#define usleep(x)   			vos_util_delay_us(x)
#include <kwrap/examsys.h> 	//for MAIN(), GETCHAR() API
#define MAIN(argc, argv) 		EXAMFUNC_ENTRY(ai_net_with_buf, argc, argv)
#endif
//#define _BSP_NA51055_
///////////////////////////////////////////////////////////////////////////////
#define VENDOR_AI_CFG  				0x000f0000  //vendor ai config
///////////////////////////////////////////////////////////////////////////////

// #include "jpeglib.h"	
#include <setjmp.h>

typedef struct my_error_mgr * my_error_ptr;

UINT32 time_model_load_init = 0;
UINT32 time_image_load_init = 0;
UINT32 time_inference = 0;
UINT32 time_postprocess = 0.0;
UINT32 time_flush = 0;
UINT32 time_transpose_data = 0;
UINT32 time_nms = 0;
UINT32 time_transfom = 0;

/*-----------------------------------------------------------------------------*/
/* Type Definitions                                                            */
/*-----------------------------------------------------------------------------*/
typedef struct _MEM_PARM {
	UINT32 pa;
	UINT32 va;
	UINT32 size;
	UINT32 blk;
} MEM_PARM;

typedef struct _NET_PROC {
	CHAR model_filename[256];
	INT32 binsize;
	int job_method;
	int job_wait_ms;
	int buf_method;
	MEM_PARM proc_mem;
	UINT32 proc_id;
	MEM_PARM io_mem;
} NET_PROC;

typedef struct _NET_IN {
	CHAR input_filename[256];
	VENDOR_AI_BUF src_img;
} NET_IN;

static HD_RESULT hdal_mem_alloc(MEM_PARM *mem_parm, CHAR* name, UINT32 size)
{
	HD_RESULT ret = HD_OK;
	UINT32 pa   = 0;
	void  *va   = NULL;

	//alloc private pool
	ret = hd_common_mem_alloc(name, &pa, (void**)&va, size, DDR_ID0);
	if (ret!= HD_OK) {
		printf("hd_common_mem_alloc (%s) size fail\n", name);
		return ret;
	}

	mem_parm->pa   = pa;
	mem_parm->va   = (UINT32)va;
	mem_parm->size = size;
	mem_parm->blk  = (UINT32)-1;
	return HD_OK;
}

static HD_RESULT hdal_mem_free(MEM_PARM *mem_parm)
{
	HD_RESULT ret = HD_OK;
	
	//free private pool
	ret =  hd_common_mem_free(mem_parm->pa, (void *)mem_parm->va);
	if (ret!= HD_OK) {
		return ret;
	}
	
	mem_parm->pa = 0;
	mem_parm->va = 0;
	mem_parm->size = 0;
	mem_parm->blk = (UINT32)-1;
	return HD_OK;
}

static INT32 _getsize_model(char* filename)
{
	FILE *bin_fd;
	UINT32 bin_size = 0;

	bin_fd = fopen(filename, "rb");
	if (!bin_fd) {
		printf("get bin(%s) size fail\n", filename);
		return (-1);
	}

	fseek(bin_fd, 0, SEEK_END);
	bin_size = ftell(bin_fd);
	fseek(bin_fd, 0, SEEK_SET);
	fclose(bin_fd);

	return bin_size;
}

static long _filesize(FILE* stream)
{
	long curpos, length;
	curpos = ftell(stream);
	fseek(stream, 0L, SEEK_END);
	length = ftell(stream);
	fseek(stream, curpos, SEEK_SET);
	return length;

}

static INT32 load_bin_file_to_mem(const CHAR *filename, MEM_PARM *mem_parm)
{
	FILE *fd;
	INT32 size = 0;
	
	fd = fopen(filename, "rb");
	if (!fd) {
		printf("cannot read %s\r\n", filename);
		return -1;
	}

	fseek(fd, 0, SEEK_END);
	size = ftell(fd);
	fseek(fd, 0, SEEK_SET);
	
	if (size < 0) {
		printf("getting %s size failed\r\n", filename);
	} else if ((INT32)fread((VOID *)mem_parm->va, 1, size, fd) != size) {
		printf("read size < %ld\r\n", size);
		size = -1;
	}
	mem_parm->size = size;
	
	if (fd) {
		fclose(fd);
	}

	return size;
}

/*-----------------------------------------------------------------------------*/
/* Network Functions                                                             */
/*-----------------------------------------------------------------------------*/
static HD_RESULT network_open(NET_PROC *p_net)
{
	HD_RESULT ret = HD_OK;
	INT32 loadsize = 0;
	
	// if (strlen(p_net->model_filename) == 0) {
	// 	printf("proc_id(%u) model is null\r\n", p_net->proc_id);
	// 	return 0;
	// }

	// hdal_mem_alloc(&p_net->proc_mem, "ai_model", p_net->binsize);
	char ai_model[16];
	snprintf(ai_model, sizeof(ai_model), "%s", "ai_model");
	hdal_mem_alloc(&p_net->proc_mem, ai_model, p_net->binsize);

	//load file
	loadsize = load_bin_file_to_mem(p_net->model_filename, &(p_net->proc_mem));
	if (loadsize <= 0) {
		printf("proc_id(%u) model load fail: %s\r\n", p_net->proc_id, p_net->model_filename);
		// return 0;
	}
	
	// set model
	vendor_ai_net_set(p_net->proc_id, VENDOR_AI_NET_PARAM_CFG_MODEL, (VENDOR_AI_NET_CFG_MODEL*)&p_net->proc_mem);
	
	// set buf opt
	VENDOR_AI_NET_CFG_BUF_OPT cfg_buf_opt;
	cfg_buf_opt.method = (VENDOR_AI_NET_BUF_OPT)p_net->buf_method;
	cfg_buf_opt.ddr_id = DDR_ID0;
	vendor_ai_net_set(p_net->proc_id, VENDOR_AI_NET_PARAM_CFG_BUF_OPT, &cfg_buf_opt);
	
	// set job opt
	VENDOR_AI_NET_CFG_JOB_OPT cfg_job_opt;
	cfg_job_opt.method  = (VENDOR_AI_NET_JOB_OPT)p_net->job_method;
	cfg_job_opt.wait_ms = p_net->job_wait_ms;
	cfg_job_opt.schd_parm = VENDOR_AI_FAIR_CORE_ALL; 	
	vendor_ai_net_set(p_net->proc_id, VENDOR_AI_NET_PARAM_CFG_JOB_OPT, &cfg_job_opt);
	
	// open
	vendor_ai_net_open(p_net->proc_id);
	
	//network_alloc_io_buf
	VENDOR_AI_NET_CFG_WORKBUF wbuf = {0};
	ret = vendor_ai_net_get(p_net->proc_id, VENDOR_AI_NET_PARAM_CFG_WORKBUF, &wbuf);
	if (ret != HD_OK) {
		printf("proc_id(%lu) get VENDOR_AI_NET_PARAM_CFG_WORKBUF fail\r\n", p_net->proc_id);
		return HD_ERR_FAIL;
	}

	char ai_io_buf[16];
	snprintf(ai_io_buf, sizeof(ai_io_buf), "%s", "ai_io_buf");
	// ret = hdal_mem_alloc(&p_net->io_mem, "ai_io_buf", wbuf.size);
	ret = hdal_mem_alloc(&p_net->io_mem, ai_io_buf, wbuf.size);
	if (ret != HD_OK) {
		printf("proc_id(%lu) alloc ai_io_buf fail\r\n", p_net->proc_id);
		return HD_ERR_FAIL;
	}

	wbuf.pa = p_net->io_mem.pa;
	wbuf.va = p_net->io_mem.va;
	wbuf.size = p_net->io_mem.size;
	ret = vendor_ai_net_set(p_net->proc_id, VENDOR_AI_NET_PARAM_CFG_WORKBUF, &wbuf);
	if (ret != HD_OK) {
		printf("proc_id(%lu) set VENDOR_AI_NET_PARAM_CFG_WORKBUF fail\r\n", p_net->proc_id);
		return HD_ERR_FAIL;
	}
	
	//printf("alloc_io_buf: work buf, pa = %#lx, va = %#lx, size = %lu\r\n", wbuf.pa, wbuf.va, wbuf.size);
	
	return ret;
}

static HD_RESULT network_close(NET_PROC *p_net)
{
	HD_RESULT ret = HD_OK;
	
	//network_free_io_buf
	if (p_net->io_mem.pa && p_net->io_mem.va) {
		hdal_mem_free(&p_net->io_mem);
	}
	
	// close
	ret = vendor_ai_net_close(p_net->proc_id);
	hdal_mem_free(&p_net->proc_mem);

	return ret;
}

int run_ai_inference(void)
{
	HD_RESULT ret = HD_OK;
	HD_COMMON_MEM_INIT_CONFIG mem_cfg;
	memset(&mem_cfg, 0, sizeof(HD_COMMON_MEM_INIT_CONFIG));
	struct timeval tstart, tend;
	INT32 proc_id = 0;
	
	// config extend engine plugin, process scheduler
	UINT32 schd = VENDOR_AI_PROC_SCHD_FAIR;
	// vendor_ai_cfg_set(VENDOR_AI_CFG_PLUGIN_ENGINE, vendor_ai_cpu1_get_engine());
	vendor_ai_cfg_set(VENDOR_AI_CFG_PROC_SCHD, &schd);
	ret = vendor_ai_init();
	if (ret != HD_OK) {
		printf("vendor_ai_init fail=%d\n", ret);
		// goto exit_thread;
	}
	
	// get vendor ai version
	VENDOR_AI_NET_CFG_IMPL_VERSION ai_version;
	ret = vendor_ai_cfg_get(VENDOR_AI_CFG_IMPL_VERSION, &ai_version);
	if (ret != HD_OK) {
		printf("get vendor ai sdk fail=%d\n", ret);
		// goto exit_thread;
	}
	printf("vendor ai sdk:\r\n");
	printf("    vendor_ai version:= %s\n", ai_version.vendor_ai_impl_version);
	printf("    kflow_ai  version:= %s\n", ai_version.kflow_ai_impl_version);
	printf("    kdrv_ai   version:= %s\n", ai_version.kdrv_ai_impl_version);
	
	//net model parameters
	NET_PROC net_info;
	net_info.proc_id = proc_id;
	sprintf(net_info.model_filename, "nvt_model.bin");
	net_info.binsize		= _getsize_model(net_info.model_filename);
	net_info.job_method     = VENDOR_AI_NET_JOB_OPT_LINEAR_O1;
	net_info.job_wait_ms    = 0;
	net_info.buf_method     = VENDOR_AI_NET_BUF_OPT_SHRINK_O1;
	if (net_info.binsize <= 0) {
		printf("proc_id(%u) model is not exist?\r\n", proc_id);
		// goto exit_thread;
	}
	//printf("proc_id(%u) set net_info: model-file(%s), binsize=%d\r\n", proc_id, net_info.model_filename, net_info.binsize);
	
	mem_cfg.pool_info[0].type = HD_COMMON_MEM_USER_DEFINIED_POOL;
	mem_cfg.pool_info[0].blk_size = 32;
	mem_cfg.pool_info[0].blk_cnt = 1;
	mem_cfg.pool_info[0].ddr_id = DDR_ID0;
	ret = hd_common_mem_init(&mem_cfg);
	if (HD_OK != ret) {
		printf("hd_common_mem_init err: %d\r\n", ret);
		// goto exit_thread;
	}
	
	gettimeofday(&tstart, NULL);  
	if ((ret = network_open(&net_info)) != HD_OK) {
		printf("proc_id(%u  nn open fail !!\n", proc_id);
		// goto exit_thread;
	}
	
	ret = vendor_ai_net_start(proc_id);
	if (HD_OK != ret) {
		printf("proc_id(%u) nn start fail !!\n", proc_id);
		// goto exit_thread;
	}
	gettimeofday(&tend, NULL);  
	time_model_load_init = (tend.tv_sec - tstart.tv_sec) * 1000000 + (tend.tv_usec - tstart.tv_usec);
	printf("ai: model load time: %d us\r\n", time_model_load_init / 1000);
		
	char filename[1024];
	UINT32 image_file_count = 0;
	UINT32 scale_out_w = 64, scale_out_h = 51;
	MEM_PARM scale_out_buf;
	input_info_t wav_info = {0};

	char scale_out_buf_[16];
	snprintf(scale_out_buf_, sizeof(scale_out_buf_), "%s", "scale_out_buf_");
	ret = hdal_mem_alloc(&scale_out_buf, scale_out_buf_, (UINT32)(scale_out_w * scale_out_h * 10));

	FILE *fp_image = fopen("wav_list.txt", "r");
	if(fp_image == NULL)
	{
		printf("wav_list.txt open fail !!\n");
		// goto exit_thread;
	}

	//input image parameters
		NET_IN nn_in;
		nn_in.src_img.width 	= scale_out_w;
		nn_in.src_img.height 	= scale_out_h;
		nn_in.src_img.channel 	= 1;
		nn_in.src_img.line_ofs 	= scale_out_w;
		nn_in.src_img.fmt      	= (HD_VIDEO_PXLFMT)0xA1100F00;
		nn_in.src_img.sign 		= MAKEFOURCC('A','B','U','F');
		nn_in.src_img.size 		= scale_out_w * scale_out_h * 1;
		nn_in.src_img.pa 		= scale_out_buf.pa;
		nn_in.src_img.va 		= scale_out_buf.va;
	
	while(!feof(fp_image))
    {
		fscanf(fp_image, "%s\n", filename);
		printf("%d file: %s\n", image_file_count, filename);
		
		// VENDOR_AI_BUF src_image;
		FILE *in_wav_file = NULL;		
		if ((in_wav_file = fopen(filename, "rb")) == NULL)
		{
			printf("error to read file %s", filename);
			// goto exit_thread;
		}

		long buffsize = _filesize(in_wav_file);
		char* buff = (char*)malloc(buffsize * sizeof(char));
		size_t ret = fread(buff, buffsize, 1, in_wav_file);
		if (ret <= 0) {
			printf("empty file to use!!\n");
		}
		fclose(in_wav_file);

		WaveInfo wavInfo;
		wavInfo.wavLen = (unsigned int)buffsize;
		wavInfo.wavData = (unsigned char*)buff;	

		gettimeofday(&tstart, NULL);

		wav_info = wyzeIDetectWave(&wavInfo);

		gettimeofday(&tend, NULL);  
		printf("wyzeIDetectWave preprocess time (us): %lu\r\n", (tend.tv_sec - tstart.tv_sec) * 1000000 + (tend.tv_usec - tstart.tv_usec));
	
		// set input image and do net proc
		memcpy((void*)nn_in.src_img.va, wav_info.image, sizeof(short)*wav_info.w*wav_info.h);
		hd_common_mem_flush_cache((VOID *)scale_out_buf.va, sizeof(short)*wav_info.w*wav_info.h);

		ret = vendor_ai_net_set(proc_id, (VENDOR_AI_NET_PARAM_ID)VENDOR_AI_NET_PARAM_IN(0, 0), &(nn_in.src_img));
		// ret = vendor_ai_net_set(proc_id,(VENDOR_AI_NET_PARAM_ID) 0, &(nn_in.src_img));
		if (HD_OK != ret) {
			printf("proc_id(%u) push input fail !!\n", proc_id);
		}

		ret = vendor_ai_net_proc(proc_id);
		if (HD_OK != ret) {
			printf("proc_id(%u) do net proc fail !!\n", proc_id);
		}
		gettimeofday(&tend, NULL);  
		time_inference += (tend.tv_sec - tstart.tv_sec) * 1000000 + (tend.tv_usec - tstart.tv_usec);
		
		image_file_count++;
		free(buff);
	}
	fclose(fp_image);
	if(wav_info.image != NULL){
		free(wav_info.image);
		printf("free wave_info mamory!\n");
	}
	
	ret = hdal_mem_free(&scale_out_buf);
	
// exit_thread:
	// stop
	ret = vendor_ai_net_stop(proc_id);
	if (HD_OK != ret) {
		printf("proc_id(%u) nn stop fail !!\n", proc_id);
	}
	
	// close network modules
	if ((ret = network_close(&net_info)) != HD_OK) {
		printf("proc_id(%u) nn close fail !!\n", proc_id);
	}

	// uninit network modules
	ret = vendor_ai_uninit();
	if (ret != HD_OK) {
		printf("vendor_ai_uninit fail=%d\n", ret);
	}

	return 0;
}

int main(int argc, char *argv[])
{
	HD_RESULT ret;

	ret = hd_common_init(0);
	if (ret != HD_OK) {
		printf("hd_common_init fail=%d\n", ret);
		goto exit;
	}
	hd_common_sysconfig(0, (1<<16), 0, VENDOR_AI_CFG); //enable AI engine
	
	ret = hd_gfx_init();
	if (ret != HD_OK) {
		printf("hd_gfx_init fail\r\n");
		goto exit;
	}
	
	run_ai_inference();
	
exit:
	ret = hd_gfx_uninit();
	if (ret != HD_OK) {
		printf("hd_gfx_uninit fail\r\n");
	}
	
	ret = hd_common_mem_uninit();
	if (ret != HD_OK) {
		printf("mem fail=%d\n", ret);
	}
	usleep(300000);
	ret = hd_common_uninit();
	if (ret != HD_OK) {
		printf("common fail=%d\n", ret);
	}

	return ret;
}
