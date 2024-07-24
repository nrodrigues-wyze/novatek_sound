#ifndef _VOS_SXCMD_H_
#define _VOS_SXCMD_H_

typedef BOOL (*SX_CMD)(unsigned char argc, char **argv); ///< Command handler function prototype
typedef struct _SX_CMD_ENTRY {
	CHAR    *p_name; ///< command's module name
	SX_CMD   p_func; ///< command table
	CHAR    *p_desc;
}
SX_CMD_ENTRY;

#define SXCMD_BEGIN(tbl, desc)  SX_CMD_ENTRY (tbl)[] = { {#tbl, 0, desc}, ///< begin a command table
#define SXCMD_ITEM(cmd, func, desc)  { (cmd), (func), (desc) }, ///< insert a command item in command table
#define SXCMD_END()    {0, 0, NULL} }; ///< end a command table

#define SXCMD_NUM(tbl)	(sizeof(tbl)/sizeof(SX_CMD_ENTRY)-2)

#define sxcmd_addtable(arg...)

#endif//_VOS_SXCMD_H_

