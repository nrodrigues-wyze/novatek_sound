/**
	@brief Header file of vendor videodec module.\n
	This file contains the functions which is related to vendor videodec.

	@file vendor_videodec.h

	@ingroup mhdal

	@note Nothing.

	Copyright Novatek Microelectronics Corp. 2018.  All rights reserved.
*/

#ifndef _VENDOR_VIDEODEC_H_
#define _VENDOR_VIDEODEC_H_

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
//------
typedef enum _VENDOR_VIDEODEC_JPG_SCALE_RATIO {
	VENDOR_VDODEC_SUB_RATIO_1_2,            ///< enable, w : 1/2  h : 1/2
	VENDOR_VDODEC_SUB_RATIO_1_4,            ///< enable, w : 1/4  h : 1/4
	VENDOR_VDODEC_SUB_RATIO_1_8,            ///< enable, w : 1/8  h : 1/8
	ENUM_DUMMY4WORD(VENDOR_VIDEODEC_JPG_SCALE_RATIO)
} VENDOR_VIDEODEC_JPG_SCALE_RATIO;

//------
typedef struct _VENDOR_VIDEODEC_OUT {
	UINT32 dec_status;                     ///< decoder status
} VENDOR_VIDEODEC_OUT;

//------
typedef struct _VENDOR_VIDEODEC_YUV_AUTO_DROP {
	BOOL enable;                     ///< yuv auto drop 
} VENDOR_VIDEODEC_YUV_AUTO_DROP;

//------
typedef struct _VENDOR_VIDEODEC_RAWQUE_MAX_NUM {
	UINT32 rawque_max_num;                      ///< raw que max number
} VENDOR_VIDEODEC_RAWQUE_MAX_NUM;

//------
typedef struct _VENDOR_VIDEODEC_JPG_SCALE_RATIO_CFG {
	BOOL enable;
	VENDOR_VIDEODEC_JPG_SCALE_RATIO scale_ratio;
} VENDOR_VIDEODEC_JPG_SCALE_RATIO_CFG;

//------
typedef enum _VENDOR_VIDEODEC_PARAM_ID {
	VENDOR_VIDEODEC_PARAM_IN_YUV_AUTO_DROP,      ///< CARDV only.  support set with i/o path, using VENDOR_VIDEODEC_YUV_AUTO_DROP struct
	VENDOR_VIDEODEC_PARAM_IN_RAWQUE_MAX_NUM,     ///< support set with i/o path, using VENDOR_VIDEODEC_RAWQUE_MAX_NUM struct
	VENDOR_VIDEODEC_PARAM_OUT_STATUS,            ///< CARDV only.  return decode status is 1: start/ 0: stop
	VENDOR_VIDEODEC_PARAM_OUT_JPG_SCALE_RATIO,   ///< support set with i/o path, using VENDOR_VIDEODEC_JPG_SCALE_RATIO_CFG struct
	ENUM_DUMMY4WORD(VENDOR_VIDEODEC_PARAM_ID)
} VENDOR_VIDEODEC_PARAM_ID;

/********************************************************************
	EXTERN VARIABLES & FUNCTION PROTOTYPES DECLARATIONS
********************************************************************/
HD_RESULT vendor_videodec_set(HD_PATH_ID path_id, VENDOR_VIDEODEC_PARAM_ID id, VOID *p_param);
HD_RESULT vendor_videodec_get(HD_PATH_ID path_id, VENDOR_VIDEODEC_PARAM_ID id, VOID *p_param);
#endif

