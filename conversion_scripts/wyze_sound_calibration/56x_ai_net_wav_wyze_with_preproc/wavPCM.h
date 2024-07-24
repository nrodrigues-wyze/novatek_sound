#pragma once
#include "common.h"

/* Alaw
* Linear Input Code Compressed Code
* ------------------------ ---------------
* 0000000wxyza   000wxyz
* 0000001wxyza   001wxyz
* 000001wxyzab   010wxyz
* 00001wxyzabc   011wxyz
* 0001wxyzabcd   100wxyz
* 001wxyzabcde   101wxyz
* 01wxyzabcdef   110wxyz
* 1wxyzabcdefg   111wxyz
*/

unsigned char LinearToAlawSample(short sample);
short AlawToLinearSample(unsigned char sample);


/* ulaw
* Biased Linear Input Code Compressed Code
* ------------------------ ---------------
* 00000001wxyza   000wxyz
* 0000001wxyzab   001wxyz
* 000001wxyzabc   010wxyz
* 00001wxyzabcd   011wxyz
* 0001wxyzabcde   100wxyz
* 001wxyzabcdef   101wxyz
* 01wxyzabcdefg   110wxyz
* 1wxyzabcdefgh   111wxyz
*/

unsigned char LinearToMuLawSample(short sample);
short MuLawToLinearSample(unsigned char sample);
