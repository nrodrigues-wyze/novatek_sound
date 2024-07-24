/**
	@brief Header file of vendor videoenc module.\n
	This file contains the functions which is related to vendor videoenc.

	@file vendor_videoenc.h

	@ingroup mhdal

	@note Nothing.

	Copyright Novatek Microelectronics Corp. 2018.  All rights reserved.
*/

#ifndef _VENDOR_VIDEOENC_H_
#define _VENDOR_VIDEOENC_H_

/********************************************************************
	INCLUDE FILES
********************************************************************/
#include "hd_type.h"

/********************************************************************
	MACRO CONSTANT DEFINITIONS
********************************************************************/

/********************************************************************
	MACRO FUNCTION DEFINITIONS
********************************************************************/

/********************************************************************
	TYPE DEFINITION
********************************************************************/
typedef enum {
	VENDOR_VIDEOENC_PADDING_MODE_ZERO = 0,	// padding with zero (default)
	VENDOR_VIDEOENC_PADDING_MODE_COPY = 1	// padding with copied bounday-line pixels
} VENDOR_VIDEOENC_PADDING_MODE;

typedef enum {
	VENDOR_VDOENC_XVR_DISABLE = 0,          ///< set xvr application disable
	VENDOR_VDOENC_XVR_56XE_528D             ///< set xvr application for 56x enc, 528 dec
} VENDOR_VDOENC_XVR_APP;

typedef struct _VENDOR_VIDEOENC_BS_RESERVED_SIZE_CFG {
	UINT32 reserved_size;
} VENDOR_VIDEOENC_BS_RESERVED_SIZE_CFG;

//------
typedef struct _VENDOR_VIDEOENC_TIMELAPSE_TIME_CFG {
	UINT32 timelapse_time;
} VENDOR_VIDEOENC_TIMELAPSE_TIME_CFG;

//------
typedef struct _VENDOR_VIDEOENC_JPG_RC_CFG {
	UINT32 vbr_mode_en;                 // 0: cbr , 1: vbr
	UINT32 min_quality;
	UINT32 max_quality;
} VENDOR_VIDEOENC_JPG_RC_CFG;

//------
typedef struct _VENDOR_VIDEOENC_H26XENC_TRIG_SNAPSHOT {
	UINT32 phy_addr;                    ///< [w]physical address of encoded data (user provide memory space to put JPEG result)
	UINT32 size;                        ///< [w]user buffer size provided
	                                    ///< [r]real size of encoded data
	UINT32 image_quality;               ///< [w]JPEG image quality. range 1~100
	UINT64 timestamp;                   ///< [r]Encoded timestamp (unit: microsecond)
} VENDOR_VIDEOENC_H26XENC_TRIG_SNAPSHOT;

//------
typedef enum {
	VENDOR_VIDEOENC_CODEC_264 = 0,
	VENDOR_VIDEOENC_CODEC_265 = 1
} VENDOR_VIDEOENC_RDO_CODEC;

typedef struct {
	UINT8 avc_intra_4x4_cost_bias;			// default: 8, range: 0~31, //INTRA 4x4 block number decrease as the value increase
	UINT8 avc_intra_8x8_cost_bias;			// default: 8, range: 0~31, //INTRA 8x8 block number decrease as the value increase
	UINT8 avc_intra_16x16_cost_bias;		// default: 8, range: 0~31, //INTRA 16x16 block number decrease as the value increase
	UINT8 avc_inter_tu4_cost_bias;			// default: 8, range: 0~31, //INTER TU4 block number decrease as the value increase
	UINT8 avc_inter_tu8_cost_bias;			// default: 8, range: 0~31, //INTER TU8 block number decrease as the value increase
	UINT8 avc_inter_skip_cost_bias;			// default: 8, range: 0~31, //Skip mode number decrease as the value increase
} VENDOR_VIDEOENC_RDO_264;

typedef struct {
	UINT8 hevc_intra_32x32_cost_bias;		// default:  0, range: 0~15, //INTRA 32x32 block number decrease as the value increase
	UINT8 hevc_intra_16x16_cost_bias;		// default:  0, range: 0~15, //INTRA 16x16 block number decrease as the value increase
	UINT8 hevc_intra_8x8_cost_bias;			// default:  0, range: 0~15, //INTRA 8x8 block number decrease as the value increase
	INT8  hevc_inter_skip_cost_bias;		// default:  0, range: -16~15, //skip mode number decrease as the value increase
	INT8  hevc_inter_merge_cost_bias;		// default:  0, range: -16~15, //merge mode number decrease as the value increase
	UINT8 hevc_inter_64x64_cost_bias;		// default: 14, range: 0~31, //INTER 64x64 block number decrease as the value increase
	UINT8 hevc_inter_64x32_32x64_cost_bias;	// default: 28, range: 0~31, //INTER 64x32 and 32x64 block number decrease as the value increase
	UINT8 hevc_inter_32x32_cost_bias;		// default: 14, range: 0~31, //INTER 32x32 block number decrease as the value increase
	UINT8 hevc_inter_32x16_16x32_cost_bias;	// default: 28, range: 0~31, //INTER 32x16 and 16x32 block number decrease as the value increase
	UINT8 hevc_inter_16x16_cost_bias;		// default:  7, range: 0~31, //INTER 16x16 block number decrease as the value increase
} VENDOR_VIDEOENC_RDO_265;

typedef struct {
	VENDOR_VIDEOENC_RDO_CODEC		rdo_codec;	///< rdo codec. 0: h264, 1: h265
	union {
		VENDOR_VIDEOENC_RDO_264	rdo_264;        ///< parameter of h264 rdo
		VENDOR_VIDEOENC_RDO_265	rdo_265;        ///< parameter of h265 rdo
	} rdo_param;
} VENDOR_VIDEOENC_RDO_CFG;

//------
typedef struct _VENDOR_VIDEOENC_JND_CFG{
	BOOL            enable;                     ///< jnd enable. default: 0, range: 0~1 (0: disable, 1: enable)
	UINT8           str;                        ///< jnd filter strength. default: 10, range: 0~15
	UINT8           level;                      ///< jnd edge level. default: 12, range: 0~15
	UINT8           threshold;                  ///< jnd CM edge detection threshold. default: 25, range: 0~255
} VENDOR_VIDEOENC_JND_CFG;

//------
typedef struct _VENDOR_VIDEOENC_H26X_VBR_POLICY_CFG {
	UINT32 h26x_vbr_policy;
} VENDOR_VIDEOENC_H26X_VBR_POLICY_CFG;

//------
typedef struct _VENDOR_VIDEOENC_H26X_RC_GOP_CFG {
	UINT32 h26x_rc_gop;                 ///< update rc gop
} VENDOR_VIDEOENC_H26X_RC_GOP_CFG;

//------
typedef struct _VENDOR_VIDEOENC_H26X_ENC_GOP_CFG {
	UINT32 h26x_enc_gop;                ///< update enc gop & rc gop
} VENDOR_VIDEOENC_H26X_ENC_GOP_CFG;

//------
typedef struct _VENDOR_VIDEOENC_JPG_YUV_TRANS_CFG {
	UINT32 jpg_yuv_trans_en;
} VENDOR_VIDEOENC_JPG_YUV_TRANS_CFG;

//------
typedef struct _VENDOR_VIDEOENC_MIN_RATIO_CFG {
	UINT32              min_i_ratio;     ///< default: 1500 (unit : ms)
	UINT32              min_p_ratio;     ///< default: 1000 (unit : ms)
} VENDOR_VIDEOENC_MIN_RATIO_CFG;

//------
typedef struct _VENDOR_VIDEOENC_H26X_ENC_COLMV {
	UINT32              h26x_colmv_en;
} VENDOR_VIDEOENC_H26X_ENC_COLMV;

//------
typedef struct _VENDOR_VIDEOENC_H26X_COMM_RECFRM {
	UINT32              enable;
	UINT32              h26x_comm_base_recfrm_en;
	UINT32              h26x_comm_svc_recfrm_en;
	UINT32              h26x_comm_ltr_recfrm_en;
} VENDOR_VIDEOENC_H26X_COMM_RECFRM;

//------
typedef struct _VENDOR_VIDEOENC_FIT_WORK_MEMORY {
	BOOL                b_enable;
} VENDOR_VIDEOENC_FIT_WORK_MEMORY;

//------
typedef struct _VENDOR_VIDEOENC_LONG_START_CODE {
	UINT32              long_start_code_en;
} VENDOR_VIDEOENC_LONG_START_CODE;

//------
typedef struct _VENDOR_VIDEOENC_H26X_SVC_WEIGHT_MODE {
	UINT32 h26x_svc_weight_mode;
} VENDOR_VIDEOENC_H26X_SVC_WEIGHT_MODE;

//------
typedef struct _VENDOR_VIDEOENC_BS_QUICK_ROLLBACK_CFG {
	BOOL                b_enable;
} VENDOR_VIDEOENC_BS_QUICK_ROLLBACK_CFG;

//------
typedef struct _VENDOR_VIDEOENC_QUALITY_BASE_MODE {
	BOOL                quality_base_en;
} VENDOR_VIDEOENC_QUALITY_BASE_MODE;

//------
typedef struct _VENDOR_VIDEOENC_PADDING_MODE_CFG {
    VENDOR_VIDEOENC_PADDING_MODE    mode;
} VENDOR_VIDEOENC_PADDING_MODE_CFG;

//------
typedef struct _VENDOR_VIDEOENC_COMM_SRCOUT {
	BOOL                b_enable;
	BOOL                b_no_jpeg_enc;
	UINT32              addr;
	UINT32              size;
} VENDOR_VIDEOENC_COMM_SRCOUT;

//------
typedef struct _VENDOR_VIDEOENC_TIMER_TRIG_COMP {
	BOOL                b_enable;
} VENDOR_VIDEOENC_TIMER_TRIG_COMP;

//------
typedef struct _VENDOR_VIDEOENC_REQ_TARGET_I {
	BOOL                enable;           ///< request i-frame enable. default: 0, range: 0~1 (0: disable, 1: enable)
	UINT64              target_timestamp; ///< set request i-frame with target_timestamp
} VENDOR_VIDEOENC_REQ_TARGET_I;

//------
typedef struct _VENDOR_VIDEOENC_JPG_BRC_CFG {
	UINT32              upper_bound;      ///< [r/w] JPEG brc upper bound (0: use default setting)
	UINT32              lower_bound;      ///< [r/w] JPEG brc lower bound (0: use default setting)
	UINT32              brc_retry_cnt;    ///< [r/w] JPEG brc max retry count
	UINT32              quality_step;     ///< [r/w] JPEG brc quality step (0: use default setting)
	UINT32              init_quality;     ///< [r/w] JPEG init quality (0: use default setting)
} VENDOR_VIDEOENC_JPG_BRC_CFG;

//------
typedef struct _VENDOR_VIDEOENC_H26X_EVBR_STATE {
	UINT32 h26x_evbr_state;
} VENDOR_VIDEOENC_H26X_EVBR_STATE;

//------
typedef struct _VENDOR_VIDEOENC_H26X_ENC_GDR {
	BOOL                enable;           ///< enable gdr. default: 0, range: 0~1 (0: disable, 1: enable)
	UINT32              period;           ///< intra refresh period. default: 0, range: 0~0xFFFFFFFF (0: always refresh, others: intra refresh frame period)
	UINT32              number;           ///< intra refresh row number. default: 1, range: 1 ~ number of macroblock/ctu row
	BOOL                enable_gdr_i_frm; ///< gdr i-frame scheme enable. default: 0, range: 0~1 (0: disable, 1: enable)
	BOOL                enable_gdr_qp;    ///< gdr qp scheme enable. default: 0, range: 0~1 (0: disable, 1: enable)
	UINT32              gdr_qp;           ///< gdr qp, range: 0~51
} VENDOR_VIDEOENC_H26X_ENC_GDR;

//------
typedef struct _VENDOR_VIDEOENC_H26X_SEI_CHKSUM_CFG {
	BOOL                b_enable;         ///< enable bs sei check sum. default: 0, range: 0~1 (0: disable, 1: enable)
} VENDOR_VIDEOENC_H26X_SEI_CHKSUM_CFG;

//------
typedef struct _VENDOR_VIDEOENC_H26X_XVR_CFG {
	VENDOR_VDOENC_XVR_APP xvr_app;        ///< xvr application
} VENDOR_VIDEOENC_H26X_XVR_CFG;

//------
typedef struct _VENDOR_VIDEOENC_H26X_SKIP_FRM_CFG {
	BOOL                b_enable;         ///< enable skip frame. default: 0, range: 0~1 (0: disable, 1: enable)
	UINT32              target_fr;        ///< target frame rate
	UINT32              input_frm_cnt;    ///< real input frame number
} VENDOR_VIDEOENC_H26X_SKIP_FRM_CFG;

//------
typedef struct _VENDOR_VIDEOENC_BR_TOLERANCE {
	UINT32              br_tolerance;    ///< default: 2, range: 0~6
} VENDOR_VIDEOENC_BR_TOLERANCE;

//------
typedef struct _VENDOR_VIDEOENC_H26X_CROP_CFG {
	BOOL                enable;           ///< crop enable
	UINT16              disp_w;           ///< display width
	UINT16              disp_h;           ///< display height
} VENDOR_VIDEOENC_H26X_CROP_CFG;

//------
typedef struct _VENDOR_VIDEOENC_H26X_LOW_POWER_CFG {
	BOOL                b_enable;         ///< enable H26x low power. default: 0, range: 0~1 (0: disable, 1: enable)
} VENDOR_VIDEOENC_H26X_LOW_POWER_CFG;

//------
typedef struct _VENDOR_VIDEOENC_OSG_QP {
	UINT8               lpm_mode;
	UINT8               tnr_mode;
	UINT8               fro_mode;
} VENDOR_VIDEOENC_OSG_QP;

//------
typedef struct _VENDOR_VIDEOENC_OSG_COLOR_INVERT {
	UINT8               blk_num;
	UINT8               org_color_level;
	UINT8               inv_color_level;
	UINT8               nor_diff_th;
	UINT8               inv_diff_th;
	UINT8               sta_only_mode;
	UINT8               full_eval_mode;
	UINT8               eval_lum_targ;
} VENDOR_VIDEOENC_OSG_COLOR_INVERT;

//------
typedef struct _VENDOR_VIDEOENC_OSG_MASK {
	HD_OSG_MASK_TYPE    type;
	UINT32              color;
	UINT32              alpha;
	HD_UPOINT           position;
	HD_DIM              dim;
	UINT32              thickness;
	UINT32              layer;
	UINT32              region;
} VENDOR_VIDEOENC_OSG_MASK;

//------
typedef struct _VENDOR_VIDEOENC_DMA_ABORT {
	UINT32              dma_abort_en;
} VENDOR_VIDEOENC_DMA_ABORT;

//------
typedef struct _VENDOR_VIDEOENC_BS_QUEUE_CNT_CFG {
	UINT32 queue_cnt;
} VENDOR_VIDEOENC_BS_QUEUE_CNT_CFG;

//------
typedef enum _VENDOR_VIDEOENC_PARAM_ID {
	VENDOR_VIDEOENC_PARAM_OUT_BS_RESERVED_SIZE,   ///< IPC only.  support set with i/o path, using VENDOR_VIDEOENC_BS_RESERVED_SIZE_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_TIMELAPSE_TIME,     ///< IPC only.  support set with i/o path, using VENDOR_VIDEOENC_TIMELAPSE_TIME_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_JPG_RC,             ///< IPC only.  support set with i/o path, using VENDOR_VIDEOENC_JPG_RC_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_TRIG_SNAPSHOT,      ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26XENC_TRIG_SNAPSHOT struct
	VENDOR_VIDEOENC_PARAM_OUT_RDO,                ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_RDO_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_JND,                ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_JND_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_H26X_VBR_POLICY,    ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_VBR_POLICY_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_H26X_RC_GOP,        ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_RC_GOP_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_H26X_ENC_GOP,       ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_ENC_GOP_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_JPG_YUV_TRANS,      ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_JPG_YUV_TRANS_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_MIN_RATIO,          ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_MIN_RATIO_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_COLMV,              ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_ENC_COLMV struct
	VENDOR_VIDEOENC_PARAM_OUT_FIT_WORK_MEMORY,    ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_FIT_WORK_MEMORY struct
	VENDOR_VIDEOENC_PARAM_OUT_LONG_START_CODE,    ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_LONG_START_CODE struct
	VENDOR_VIDEOENC_PARAM_OUT_H26X_SVC_WEIGHT_MODE,    ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_SVC_WEIGHT_MODE struct
	VENDOR_VIDEOENC_PARAM_OUT_QUALITY_BASE,       ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_QUALITY_BASE_MODE struct
	VENDOR_VIDEOENC_PARAM_OUT_DMA_ABORT,          ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_DMA_ABORT struct
	VENDOR_VIDEOENC_PARAM_OUT_COMM_RECFRM,		  ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_COMM_RECFRM struct
	VENDOR_VIDEOENC_PARAM_OUT_BS_QUICK_ROLLBACK,  ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_BS_QUICK_ROLLBACK_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_PADDING_MODE,       ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_PADDING_MODE_CFG struct
	VENDOR_VIDEOENC_PARAM_IN_STAMP_QP,            ///< IPC only.  support set with i/o path, using VENDOR_VIDEOENC_OSG_QP struct
	VENDOR_VIDEOENC_PARAM_IN_STAMP_COLOR_INVERT,  ///< IPC only.  support set with i/o path, using VENDOR_VIDEOENC_OSG_COLOR_INVERT struct
	VENDOR_VIDEOENC_PARAM_IN_MASK_ATTR,           ///< IPC only.  support set with i/o path, using VENDOR_VIDEOENC_OSG_MASK struct
	VENDOR_VIDEOENC_PARAM_OUT_BS_QUEUE_CNT,       ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_BS_QUEUE_CNT_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_COMM_SRCOUT,        ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_COMM_SRCOUT struct
	VENDOR_VIDEOENC_PARAM_OUT_TIMER_TRIG_COMP,    ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_TIMER_TRIG_COMP struct
	VENDOR_VIDEOENC_PARAM_OUT_REQ_TARGET_I,       ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_REQ_TARGET_I struct
	VENDOR_VIDEOENC_PARAM_OUT_JPG_BRC,            ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_JPG_BRC_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_H26X_EVBR_STATE,	  ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_EVBR_STATE struct
	VENDOR_VIDEOENC_PARAM_OUT_H26X_ENC_GDR,       ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_ENC_GDR struct
	VENDOR_VIDEOENC_PARAM_OUT_H26X_SEI_CHKSUM,    ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_SEI_CHKSUM_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_H26X_XVR,           ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_XVR_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_H26X_SKIP_FRM,      ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_SKIP_FRM_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_BR_TOLERANCE,       ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_BR_TOLERANCE struct
	VENDOR_VIDEOENC_PARAM_OUT_H26X_LOW_POWER,     ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_LOW_POWER_CFG struct
	VENDOR_VIDEOENC_PARAM_OUT_H26X_CROP,          ///< IPC only   support set with i/o path, using VENDOR_VIDEOENC_H26X_CROP_CFG struct
	ENUM_DUMMY4WORD(VENDOR_VIDEOENC_PARAM_ID)
} VENDOR_VIDEOENC_PARAM_ID;

/********************************************************************
	EXTERN VARIABLES & FUNCTION PROTOTYPES DECLARATIONS
********************************************************************/
HD_RESULT vendor_videoenc_set(HD_PATH_ID path_id, VENDOR_VIDEOENC_PARAM_ID id, VOID *p_param);
HD_RESULT vendor_videoenc_get(HD_PATH_ID path_id, VENDOR_VIDEOENC_PARAM_ID id, VOID *p_param);
#endif

