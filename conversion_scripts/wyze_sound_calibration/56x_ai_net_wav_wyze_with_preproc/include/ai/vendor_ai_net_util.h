/**
	@brief Header file of utility functions of vendor net sample.

	@file net_util_sample.h

	@ingroup net_util_sample

	@note Nothing.

	Copyright Novatek Microelectronics Corp. 2018.  All rights reserved.
*/
#ifndef _VENDOR_AI_NET_UTIL_H_
#define _VENDOR_AI_NET_UTIL_H_

/********************************************************************
	INCLUDE FILES
********************************************************************/
#include "hd_type.h"
#include "kwrap/nvt_type.h"

/********************************************************************
	MACRO CONSTANT DEFINITIONS
********************************************************************/
#define NET_UTIL_MODULE_VERSION 	"1.00.002"

#undef MIN
#define MIN(a, b)           ((a) < (b) ? (a) : (b))
#undef MAX
#define MAX(a, b)           ((a) > (b) ? (a) : (b))
#undef ABS
#define ABS(a)              ((a) >= 0 ? (a) : (-a))
#undef CLAMP
#define CLAMP(x,min,max)    (((x) > (max)) ? (max) : (((x) > (min)) ? (x) : (min)))

#undef SWAP
#define SWAP(a, b, t)       (t) = (a); (a) = (b); (b) = (t)

/********************************************************************
	EXTERN VARIABLES & FUNCTION PROTOTYPES DECLARATIONS
********************************************************************/

#endif  /* _VENDOR_AI_NET_UTIL_H_ */
