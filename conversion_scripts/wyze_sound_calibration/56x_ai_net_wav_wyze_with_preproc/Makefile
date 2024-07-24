MODULE_NAME = demo
# DIRs

HDAL_LIB = ./include/hdal
VOSL_LIB = ./include/vos
AI_LIB = ./include/ai

.PHONY: all clean
###############################################################################
# Linux Makefile                                                              #
###############################################################################
ifeq ($(NVT_PRJCFG_CFG),Linux)
#--------- ENVIRONMENT SETTING --------------------
WARNING		= -Wall -Wundef -Wsign-compare -Wno-missing-braces
COMPILE_OPTS =  -I. -O2 -fPIC -ffunction-sections -fdata-sections -D__LINUX -ftree-vectorize
C_CFLAGS	 = $(PLATFORM_CFLAGS) $(COMPILE_OPTS) $(WARNING) $(EXTRA_INCLUDE) -I$(VOSL_LIB) -I$(HDAL_LIB) -I$(AI_LIB)
LD_FLAGS    += -L$(VOSL_LIB) -lvos 
LD_FLAGS   	+= -L$(HDAL_LIB) -lpthread -lm -lhdal
LD_FLAGS 	+= -L$(HDAL_LIB) -lvendor_media
LD_FLAGS 	+= -L$(AI_LIB) -lvendor_ai2 -lvendor_ai2_pub -lprebuilt_ai

#--------- END OF ENVIRONMENT SETTING -------------
LIB_NAME = $(MODULE_NAME)
SRC = wavPCM.cpp Handlewave.cpp ai_detect_wav.cpp

OBJ = $(SRC:.cpp=.o)

ifeq ("$(wildcard *.cpp */*.cpp)","")
all:
	@echo "nothing to be done for '$(OUTPUT_NAME)'"
clean:
	@echo "nothing to be done for '$(OUTPUT_NAME)'"
else
all: $(LIB_NAME)

$(LIB_NAME): $(OBJ)
	@echo Creating $@...
	@$(CXX) -o $@ $(OBJ) $(LD_FLAGS)
	@$(NM) -n $@ > $@.sym
	@$(STRIP) $@
	@$(OBJCOPY) -R .comment -R .note.ABI-tag -R .gnu.version $@

%.o: %.cpp
	@echo Compiling $<
	@$(CXX) $(C_CFLAGS) -c $< -o $@

clean:
	@rm -f $(LIB_NAME) $(OBJ) $(LIB_NAME).sym *.o *.a *.so*
endif

install:
	@cp -avf $(LIB_NAME) $(ROOTFS_DIR)/rootfs/usr/bin


endif
