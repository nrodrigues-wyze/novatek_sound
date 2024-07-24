/**
	@brief Header file of definition of vendor user-space net flow sample.

	@file vendor_ai_plugin.h

	@ingroup vendor_ai_plugin

	@note Nothing.

	Copyright Novatek Microelectronics Corp. 2018.  All rights reserved.
*/
#ifndef _VENDOR_AI_PLUGIN_H_
#define _VENDOR_AI_PLUGIN_H_

/********************************************************************
	INCLUDE FILES
********************************************************************/
#include "hd_type.h"

/********************************************************************
	MACRO CONSTANT DEFINITIONS
********************************************************************/

/**
	Command: general control flow
*/
#define VENDOR_AI_CTRL_NET          0xff000000
#define VENDOR_AI_CTRL_LYR          0xff000000

/**
	Command: device control flow
*/
#define VENDOR_AI_CTRL_ENG_INIT     0xf0000000
#define VENDOR_AI_CTRL_ENG_UNINIT   0xf1000000

/**
	Command: network control flow
*/
#define VENDOR_AI_CTRL_NET_OPEN     0xf4000000
#define VENDOR_AI_CTRL_NET_START    0xf5000000
#define VENDOR_AI_CTRL_NET_STOP     0xf6000000
#define VENDOR_AI_CTRL_NET_CLOSE    0xf7000000

/**
	Command: layer control flow
*/
#define VENDOR_AI_CTRL_LYR_START    0xfa000000
#define VENDOR_AI_CTRL_LYR_STOP     0xfb000000



/**
	Command: get/set device param
*/
#define VENDOR_AI_PLUGIN_VER        0x0201
#define VENDOR_AI_PLUGIN_SUBVER     0x0202
#define VENDOR_AI_PLUGIN_SIGN       0x0203 // for NN_DLI operation

/**
	Command: get/set network param
*/
#define VENDOR_AI_PLUGIN_P1         0x0101

/**
	Command: get/set layer param
*/
#define VENDOR_AI_PLUGIN_BUFADDR    0
#define VENDOR_AI_PLUGIN_BUFTYPE    1
#define VENDOR_AI_PLUGIN_BUFSIZE    2
#define VENDOR_AI_PLUGIN_RESULT     3
#define VENDOR_AI_PLUGIN_BUF_PA     4

/********************************************************************
	TYPE DEFINITION
********************************************************************/

typedef HD_RESULT (*VENDOR_AI_ENG_CB)(UINT32 proc_id, UINT32 job_id);

/**
	Callback function to do layer processing
*/
typedef HD_RESULT (*VENDOR_AI_NET_PROC_CB)(UINT32 proc_id, UINT32 layer_id, UINT32 mode, UINT32 layer_param, UINT32 parm_addr);

/**
	Callback function to get device param, network param & layer param
*/
typedef HD_RESULT (*VENDOR_AI_NET_GET_CB)(UINT32 proc_id, UINT32 layer_id, UINT32 mode, UINT32 layer_param, UINT32 parm_addr, UINT32 cmd, UINT32* buf_addr, UINT32* buf_size);

/**
	Callback function to set device param, network param & layer param, and do device control flow, network control flow & layer control flow
*/
typedef HD_RESULT (*VENDOR_AI_NET_SET_CB)(UINT32 proc_id, UINT32 layer_id, UINT32 mode, UINT32 layer_param, UINT32 parm_addr, UINT32 cmd, UINT32 buf_addr, UINT32 buf_size);

/**
	Parameters of extend engine
*/
typedef struct _VENDOR_AI_ENGINE_PLUGIN {
	UINT32 sign;          ///< signature = MAKEFOURCC('A','E','N','G')
	UINT32 eng;           ///< engine id: 0=dla, 1=cpu, 2=dsp
	UINT32 ch;            ///< channel id: 0~4
	VENDOR_AI_NET_PROC_CB proc_cb;  ///< callback function to do device control flow, network control flow, layer control flow, layer processing
	VENDOR_AI_NET_GET_CB get_cb;  ///< callback function to get device param, network param & layer param
	VENDOR_AI_NET_SET_CB set_cb;  ///< callback function to set device param, network param & layer param
} VENDOR_AI_ENGINE_PLUGIN;




/********************************************************************
	EXTERN VARIABLES & FUNCTION PROTOTYPES DECLARATIONS
********************************************************************/


#endif  /* _VENDOR_AI_PLUGIN_H_ */
