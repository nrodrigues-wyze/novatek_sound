/**
	@brief Header file of vendor videoprocess module.\n
	This file contains the functions which is related to vendor videoprocess.

	@file vendor_videoprocess.h

	@ingroup mhdal

	@note Nothing.

	Copyright Novatek Microelectronics Corp. 2020.  All rights reserved.
*/

#ifndef _VENDOR_VIDEOPROCESS_H_
#define _VENDOR_VIDEOPROCESS_H_

/********************************************************************
	INCLUDE FILES
********************************************************************/
#include "hd_type.h"
#include "hd_common.h"
#include "hd_videoprocess.h"
/********************************************************************
	MACRO CONSTANT DEFINITIONS
********************************************************************/
#define HD_VIDEOPROC_FUNC_DIRECT_SCALEUP 0x04000000 ///< enable scaling up in direct mode
/********************************************************************
	MACRO FUNCTION DEFINITIONS
********************************************************************/

/********************************************************************
	TYPE DEFINITION
********************************************************************/

typedef struct _VENDOR_VIDEOPROC_SLICE_MODE {
	UINT8 enable[HD_VP_MAX_OUT];                            ///< [in]  enable slice push on some path output
	UINT8 cnt;                                              ///< [in]  slice count
} VENDOR_VIDEOPROC_SLICE_MODE;

typedef struct _VENDOR_VIDEOPROC_OUT_ONEBUF_MAX {
	UINT32 max_size;                         ///< output max size. onebuf allocate max buf
} VENDOR_VIDEOPROC_OUT_ONEBUF_MAX;

typedef enum _VENDOR_VIDEOPROC_PARAM_ID {
	VENDOR_VIDEOPROC_PARAM_HEIGHT_ALIGN,    ///< using device id, config output height align,only more buffer
	VENDOR_VIDEOPROC_PARAM_IN_DEPTH,	  	///< using in id
	VENDOR_VIDEOPROC_PARAM_DMA_ABORT,		///< using device id, do DMA abort.
	VENDOR_VIDEOPROC_PARAM_SLICE_MODE,		///< set with ctrl path, enable slice mode.
	VENDOR_VIDEOPROC_PARAM_USER_CROP_TRIG,  ///< using path id, trigger user crop flow
	VENDOR_VIDEOPROC_CFG_DIS_SCALERATIO,	///< config scale-ratio of DIS func (1100, 1200, 1400)
	VENDOR_VIDEOPROC_CFG_DIS_SUBSAMPLE,		///< config sub-sample of DIS func (0, 1, 2)
	VENDOR_VIDEOPROC_PARAM_STRIP,		    ///< set with ctrl path, set strip level.
	VENDOR_VIDEOPROC_PARAM_OUT_ONEBUF_MAX,  ///< using out path, using VENDOR_VIDEOPROC_OUT_ONEBUF_MAX struct (output frame max dim)
	VENDOR_VIDEOPROC_PARAM_DIS_CROP_ALIGN,  ///< using out path, config dis crop addr align
	VENDOR_VIDEOPROC_PARAM_LINEOFFSET_ALIGN,  ///< using out path, config line offset align
	ENUM_DUMMY4WORD(VENDOR_VIDEOPROC_PARAM_ID)
} VENDOR_VIDEOPROC_PARAM_ID;
/********************************************************************
	EXTERN VARIABLES & FUNCTION PROTOTYPES DECLARATIONS
********************************************************************/
HD_RESULT vendor_videoproc_set(UINT32 id, VENDOR_VIDEOPROC_PARAM_ID param_id, VOID *p_param);
#endif

