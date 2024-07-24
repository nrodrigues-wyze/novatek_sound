#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <unistd.h>
#include <time.h>

#include "Handlewave.h"


long _filesize(FILE* stream);


int detect(char* filename)
{
    time_t tt = time(NULL);

    printf("file:%s  %lu\n", filename, tt);
	FILE* fp;
	if ((fp = fopen(filename, "rb")) == NULL) {
		printf("error to read file %s", filename);
		return 0;
	}
	long buffsize = _filesize(fp);
	char* buff = (char*)malloc(buffsize * sizeof(char));
	size_t ret =fread(buff, buffsize, 1, fp);
	if (ret <= 0) {
		printf("empty file to use!!\n");
	}
	fclose(fp);

	WaveInfo wavInfo;
	wavInfo.wavLen = (unsigned int)buffsize;
	wavInfo.wavData = (unsigned char*)buff;	
	
	wyzeIDetectWave( &wavInfo);
}

long _filesize(FILE* stream)
{
	long curpos, length;
	curpos = ftell(stream);
	fseek(stream, 0L, SEEK_END);
	length = ftell(stream);
	fseek(stream, curpos, SEEK_SET);
	return length;
}


int main(int argc, char* argv[])
{
	char *file_path = "sound.wav";
	printf("demo is begin....\n");
    detect(file_path);
	return 0;
}