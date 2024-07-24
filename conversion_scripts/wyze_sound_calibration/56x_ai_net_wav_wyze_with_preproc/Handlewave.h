#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory> 
#include <vector>
#include <string.h>

#include <bits/stdc++.h>

#define  WAVE_HEAD_LEN  44
using namespace std;


struct wav_reader_ex {

	long data_pos;
	uint32_t data_length;
	uint32_t data_left;

	int format;
	int sample_rate;
	int bits_per_sample;
	int channels;
	int byte_rate;
	int block_align;

	int streamed;
};

enum ResultTypeWave
{
    NONE_WAVE=0,
    DOUBT_WAVE=1,
    CONFIDENT_WAVE=2
};

enum class AUDIO_TYPE
{
	AUDI0_PCM = 1,
	AUODIO_A_LAW = 6,
	AUDIO_MU_LAW = 7,	
};

typedef struct
{   
    float score;   
    int class_id;
    enum ResultTypeWave result_type;
    float low_threshold;
    float high_threshold;
    int sample_rate;
} DetectionWave;

typedef struct
{
    unsigned char* wavData;
    unsigned int wavLen;
} WaveInfo;


typedef struct
{
    short* image;
    int w;
    int h;
}input_info_t;


input_info_t wyzeIDetectWave(WaveInfo* wavInfo);

class CWaveMsg {
        CWaveMsg();

    public:
        CWaveMsg(unsigned char* wavData, unsigned int wavLen);
        CWaveMsg(void *wr);
        ~CWaveMsg();
        bool preHandleWave();
        void checkPcmType(std::vector<int16_t>& vecData, int& len, unsigned char* data, unsigned int length, int type);
        void preprocessWave(std::uint32_t & length, input_info_t* input_src, int n_fft, int n_hop, int n_mel, int fmin, int fmax, int sr);
        void forwardpass(input_info_t input_src);

    private:
    	bool extractWavHead();
    	int wav_get_header_ex(int* format, int* channels, int* sample_rate, int* bits_per_sample, unsigned int* data_length);
        int wav_read_data_ex(unsigned char* data, unsigned int length);
        uint32_t read_tag_ex(int i);
        uint32_t read_tag(int& i);
        uint32_t read_int32(int& i);
        uint16_t read_int16(int& i);
        void skip(int& pos, int n);
        std::unique_ptr<wav_reader_ex> m_raw_buffer;
        wav_reader_ex m_wave_reader;
        unsigned char* m_wavData;
        unsigned int m_wavLen;

        int m_sr;
        std::vector<float> m_x;
        std::vector<float> m_mfcc_feature;

        void* m_wr; 
};

