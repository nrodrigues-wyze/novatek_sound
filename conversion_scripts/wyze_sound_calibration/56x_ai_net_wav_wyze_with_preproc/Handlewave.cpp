
#include "Handlewave.h"
#include "wavPCM.h"
#include "librosa/librosa.h"

#ifdef __cplusplus
extern "C" {
#endif
	#include "vendor_ai_util.h"
#ifdef __cplusplus
}
#endif

#define TAG(a, b, c, d) (((a) << 24) | ((b) << 16) | ((c) << 8) | (d))


// int wyzeIDetectWave(WaveInfo* wavInfo)
input_info_t wyzeIDetectWave(WaveInfo* wavInfo)
{
	auto wyzeWaveMsg(make_unique<CWaveMsg>(wavInfo->wavData, wavInfo->wavLen));
	wyzeWaveMsg->preHandleWave();
	// auto& weights = m_weight_info->weights;
    int n_fft = 1024.0;
    int n_hop = 320.0;
    int n_mel = 64.0;
    int fmin = 20.0;
    int fmax = 14000.0;
    int sr = 16000.0;
    input_info_t input_src;
    memset(&input_src, 0, sizeof(input_info_t));

    std::uint32_t length = 0;
    wyzeWaveMsg->preprocessWave(length, &input_src, n_fft, n_hop, n_mel, fmin, fmax, sr);
    // printf("%d %d \n", input_src.w, input_src.h);

    // wyzeWaveMsg->forwardpass(input_src);

    return input_src;
}
void CWaveMsg::skip(int& pos, int n)
{
    pos += n;   
}
CWaveMsg::CWaveMsg(unsigned char* wavData, unsigned int wavLen)
{
    m_raw_buffer = make_unique<wav_reader_ex>();
    m_wr = NULL;
    m_wavData = wavData;
    m_wavLen = wavLen; 
    m_wave_reader.streamed = 0;
}

CWaveMsg::CWaveMsg(void* wr)
{
    m_raw_buffer = make_unique<wav_reader_ex>();
    m_wr = wr;
    m_wavData = NULL;
    m_wavLen = 0;
    m_wave_reader.streamed = 0;
}

CWaveMsg::~CWaveMsg()
{
    if (m_wr) {
        free(m_wr);
        m_wr = NULL;
    }
    
}

void CWaveMsg::checkPcmType(std::vector<int16_t>& vecData, int& len, unsigned char* data, unsigned int length, int type)
{
    std::vector<int16_t>  vecDecode;
     
    if ((int)AUDIO_TYPE::AUODIO_A_LAW == type) {
        for (unsigned int i = 0; i < length; i++) {
            
            vecDecode.push_back(AlawToLinearSample(data[i]));
        }
    }
    else if ((int)AUDIO_TYPE::AUDIO_MU_LAW == type) {
        for (unsigned int i = 0; i < length; i++) {
            vecDecode.push_back(MuLawToLinearSample(data[i]));
        }
    }
    if (!vecDecode.empty()) {
        vecData.clear();
        len = static_cast<int>(vecDecode.size());
        // log_i("copied vector of length %d", len);
        vecData = vecDecode;
    }
    return;
}


bool CWaveMsg::extractWavHead()
{

    if (WAVE_HEAD_LEN >= m_wavLen) {
        return false;
    }
    bool bFind = false;
    int pos = 0;
    int nCheckPos = (int)(m_wavLen - 4);
    int nSecondPos = (int)(m_wavLen - 8);

    uint32_t tag, tag2, length;
    for (;;) {
        for (;;) {
           
            if (pos >= nCheckPos) {
                string strErr = "not to find RIFF flag";
                printf("%s \n", strErr.c_str());
                return false;
            }
            
            tag = read_tag(pos);
            
            if (tag != TAG('R', 'I', 'F', 'F')) {
                continue;
            }
           
            break;
        }
       
        if (pos >= nSecondPos) {

            string strErr = "not to find WAVE flag";
            printf("%s \n", strErr.c_str());
            return false;
        }
        length = read_int32(pos);
        tag2 = read_tag(pos);
        if (tag2 == TAG('W', 'A', 'V', 'E')) {
            if ((m_wavLen-8) >= length)
                bFind = true;

            break;
        }
        
    }

    length -= 4;
    if (!bFind) {
        string strErr = "head of wave file is wrong";
        printf("%s \n", strErr.c_str());
        return false;
    }

    wav_reader_ex* wr = &m_wave_reader;
    uint32_t subtag, sublength;
    subtag = read_tag(pos);  
    if (pos >= (int)m_wavLen) {
        string strErr = "error to find ending";
        printf("%s \n", strErr.c_str());
        return false;
    }
    sublength = read_int32(pos);  
    if (subtag != TAG('f', 'm', 't', ' ')) {
        string strErr = "error to fmt ending";
        printf("%s \n", strErr.c_str());
        return false;        
    }

    wr->format = read_int16(pos);
    wr->channels = read_int16(pos);
    wr->sample_rate = read_int32(pos);
    wr->byte_rate = read_int32(pos);
    wr->block_align = read_int16(pos);
    wr->bits_per_sample = read_int16(pos);

    int offset = 0;
    if (wr->format == 0xfffe) {
        skip(pos, 8);
        wr->format = read_int32(pos);
        offset = sublength - 28;
    }
    else {
        offset = sublength - 16;        
    }

    if (offset > 0)
        skip(pos, offset);

    m_raw_buffer = make_unique<wav_reader_ex>();
    *m_raw_buffer = *wr;

    bFind = false;
    int last_pos = (int)m_wavLen - pos;
    last_pos -= 8;
    while (pos < last_pos) {
        subtag = read_tag_ex(pos);
        if (subtag == TAG('d', 'a', 't', 'a')) {
            pos += 4;
            sublength = read_int32(pos);
            wr->data_pos = pos;
            wr->data_length = sublength;
            wr->data_left = wr->data_length;         
            wr->streamed = 1;

            bFind = true;
            return bFind;
        }
        else {
            pos++;
        }
    }
     
    return bFind;
}

bool CWaveMsg::preHandleWave()
{
    bool bResult = true;
    if(!extractWavHead()) {
        string strErr = "it's error to extract wave information...\n";
        printf("%s \n", strErr.c_str());
        return false;
    }     
    int format, channels, sr, bits_per_sample;
    unsigned int data_length;
    int res = wav_get_header_ex(&format, &channels, &sr, &bits_per_sample, &data_length);
    if (!res) {       
        string strErr;
        printf("%s \n", strErr.c_str());
        return false;
    }  
    printf("audio_type:%d   data_length:%d sampling rate:%d , bits_per_sample %d\n", format, data_length, sr, bits_per_sample);


    int samples = data_length * 8 / bits_per_sample;
    // log_i("Initial number of samples %d\n", samples);
    std::vector <int16_t> tmp(samples);
    res = wav_read_data_ex(reinterpret_cast<unsigned char*>(tmp.data()), data_length);
    if (res < 0) {
        string strErr;
        printf("%s \n", strErr.c_str());
        return false;
    }

    checkPcmType(tmp, samples, reinterpret_cast<unsigned char*>(tmp.data()), data_length, format);    

    m_x.resize(samples);    
    std::vector<float>& x = m_x;
    std::transform(tmp.begin(), tmp.end(), x.begin(),[](int16_t a) {return static_cast<float>(a) / 32767.f;});    
    
    return bResult;
}

void CWaveMsg::preprocessWave(std::uint32_t& length, input_info_t* input_src, int n_fft, int n_hop, int n_mel, int fmin, int fmax, int sr)
{     
    std::vector<std::vector<float>> logmel_feature;
    vector<float> subvector = {m_x.begin(), m_x.begin() + sr*2};
    // log_i("Print length %d", length);
    
	// struct timeval tstart, tend;
	// gettimeofday(&tstart, NULL);
    
    //logmel_feature = librosa::Feature::stft(subvector, 1024, 320, "haan", true, "reflect");
    logmel_feature = librosa::Feature::melspectrogram(subvector, sr, n_fft, n_hop,"haan", true, "reflect", 2, n_mel, fmin, fmax);
	
	// gettimeofday(&tend, NULL);  
	// printf("librosa::Feature::melspectrogram (us): %lu\r\n", (tend.tv_sec - tstart.tv_sec) * 1000000 + (tend.tv_usec - tstart.tv_usec));
	
    int channels = 1;
    int width = logmel_feature[0].size();
    int height = logmel_feature.size();
    // log_i("print logmel");


    size_t size_data = height * width * channels;
    length = static_cast<uint32_t>(size_data);

    // unsigned char *sounddata = (unsigned char *)malloc(size_data * sizeof(unsigned char));
    float *sounddata = (float *)malloc(size_data * sizeof(float));
    short *sounddata_16bit = (short *)malloc(size_data * sizeof(short));

    if (sounddata)
    {
        memset(sounddata, 0, size_data * sizeof(float));
		
		// gettimeofday(&tstart, NULL);
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                sounddata[i * width + j] = (((max<float>(-100.0, min<float>(logmel_feature[i][j], 28.0)) + 100) / 128.0) * 255);
				// printf("%.3f  ", logmel_feature[i][j]);
                // printf("%.2f  ", sounddata[i * width + j]);
            }
			// printf("\r\n");
        }

        vendor_ai_cpu_util_float2fixed(sounddata, 1.0, (void *)sounddata_16bit, (HD_VIDEO_PXLFMT)0xA1100807, length);

        input_src->w = width;
        input_src->h = height;
        input_src->image = sounddata_16bit; 
		
		FILE *fp = fopen("mfcc_clamp_normalized_int16.bin", "wb");
		fwrite(input_src->image, 2, input_src->w * input_src->h, fp);
		fwrite((void*)sounddata, 4, size_data , fp);
		fclose(fp);
        free(sounddata);
        free(sounddata_16bit);
		
    }
    else {
        printf("it's fault to call malloc (%ld)", size_data);
        input_src->image = nullptr;
    }

}

uint32_t CWaveMsg::read_tag_ex(int i)
{
    uint32_t tag = 0;
    tag = (tag << 8) | m_wavData[i+0];
    tag = (tag << 8) | m_wavData[i+1];
    tag = (tag << 8) | m_wavData[i+2];
    tag = (tag << 8) | m_wavData[i+3];
    return tag;
}

uint32_t CWaveMsg::read_tag(int& i)
{
    uint32_t tag = 0;
    tag = (tag << 8) | m_wavData[i++];
    tag = (tag << 8) | m_wavData[i++];
    tag = (tag << 8) | m_wavData[i++];
    tag = (tag << 8) | m_wavData[i++];
    return tag;
}

uint32_t CWaveMsg::read_int32(int& i)
{    
    uint32_t value = 0;
    value |= m_wavData[i++] << 0;
    value |= m_wavData[i++] << 8;
    value |= m_wavData[i++] << 16;
    value |= m_wavData[i++] << 24;
    return value;
}

uint16_t CWaveMsg::read_int16(int& i)
{ 
    uint16_t value = 0;
    value |= m_wavData[i++] << 0;
    value |= m_wavData[i++] << 8;
    return value;
}

int CWaveMsg::wav_get_header_ex(int* format, int* channels, int* sample_rate, int* bits_per_sample, unsigned int* data_length)
{
    wav_reader_ex * wr = &m_wave_reader;
    if (format)
        *format = wr->format;
    if (channels)
        *channels = wr->channels;
    if (sample_rate)
        *sample_rate = wr->sample_rate;
    if (bits_per_sample)
        *bits_per_sample = wr->bits_per_sample;
    if (data_length)
        *data_length = wr->data_length;
    return wr->format && wr->sample_rate;
}

int CWaveMsg::wav_read_data_ex(unsigned char* data, unsigned int length)
{ 
    unsigned int data_pos = (unsigned int) m_wave_reader.data_pos;
    if ((data_pos + length) > m_wavLen) {

        string strErr;

        return -1;
    }
    wav_reader_ex* wr = &m_wave_reader;      
    memcpy(data, &m_wavData[data_pos], length);
    wr->data_left -= length;    
    return length;
}

// void CWaveMsg::forwardpass(input_info_t input_src){
//     // forward pass
// }
