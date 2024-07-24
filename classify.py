from models import *
from math import ceil
from utils.torch_utils import pad_split_input, move_data_to_device, audio_sample_pad
import time
import librosa
import json
from easydict import EasyDict as edict
import scipy.signal

class ClassificationEngine(object):

    def __init__(self, model_path, cfg):
        device = 'cuda:0'
        model = MobileNetv2(cfg, img_size=(51,64)).to(device)

        if model_path.endswith('.pt'):
            model.load_state_dict(torch.load(model_path, map_location=device)['model'])
            '''
            print("loaded model ", model_path)
            chkpt = torch.load(model_path, map_location=device)
            #import pdb; pdb.set_trace()
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            for k,v in chkpt['model'].items():
                if model.state_dict()[k].numel()==v.numel():
                    print(k)
            model.load_state_dict(chkpt['model'], strict=True)
            print("loaded previous model", model_path)
            '''
        #import pdb; pdb.set_trace()
        #self.labels = ['sound_other', 'dog_barking', 'cat_meowing', 'baby_crying', 'siren']
        with open('cfg/labels.json') as f:
            self.names = edict(json.load(f))
        self.debug_mode=True
        self.model = model
        self.classes_num = 8
        self.target_sample_rate = 16000
        self.audio_clip_length = 2
        self.sample_length = self.audio_clip_length * self.target_sample_rate
        self.device = device
        self.segment_length = 2
        self.overlap_length = 0

        self.window_size = 1024
        self.hop_size = 320
        self.fmin = 20
        self.fmax = 14000
        self.mel_bins = 64
        self.sample_rate = 16000

    def __current_time(self):
        """
        record time in million seconds
        """
        return int(round(time.time() * 1000))
    
    def classify_audio_ingenic(self, data, orig_sample_rate=8000,  number=0):
        '''
        classify a waveform type audio
        data: dict of {key: waveform}, or waveform.
              waveform (array): darray of sound amplitudes (audio_clip_length, )
        '''
        time_total_start = time_preprocessing_start = self.__current_time()
        #print(orig_sample_rate, self.target_sample_rate)
        #if number==136:
            #import pdb; pdb.set_trace()
        if isinstance(data, dict):
            audio_file = list(data.keys())[0]
            
            waveform = data[audio_file]
            prediction_dict = {audio_file: list()}
        else:
            prediction_list = []
            waveform = data
        if orig_sample_rate != self.target_sample_rate:
            gcd = np.gcd(orig_sample_rate, self.target_sample_rate)
            # there are many other ways to do resampling
            waveform = scipy.signal.resample_poly(
                waveform, self.target_sample_rate // gcd, orig_sample_rate // gcd, axis=-1)

        audio_length = len(waveform)/self.target_sample_rate
        waveform = torch.tensor(waveform[None, :])
        #if number==136:
        #    import pdb; pdb.set_trace()
        #print(waveform.shape[1])
        if waveform.shape[1] < self.sample_length:
            n_clips = 1
            n_segs = int(ceil(
                waveform.shape[1] / ((self.segment_length - self.overlap_length) * self.target_sample_rate)))
            audio_segments = pad_split_input(
                waveform, n_clips, n_segs, self.segment_length, self.overlap_length, self.target_sample_rate)
        else:
            n_clips = int(ceil(waveform.shape[1] / self.sample_length))
            audio_clips = audio_sample_pad(
                waveform, n_clips, self.sample_length)
            n_segs = int(ceil(self.sample_length / ((self.segment_length -
                         self.overlap_length) * self.target_sample_rate)))
            audio_segments = pad_split_input(
                audio_clips, n_clips, n_segs, self.segment_length, self.overlap_length, self.target_sample_rate)

        all_audio_segments=[]
        for kkk in range(audio_segments.shape[0]):
            #all_audio_segments.append(librosa.feature.mfcc(y=np.array(audio_segments)[kkk], sr=8000,
            #                                  n_mfcc=64, dct_type=2, norm='ortho',
            #                                  n_mels=64, hop_length=320,
            #                                  fmin=20, fmax=14000, n_fft=1024).transpose().astype('float32'))
            single_data=np.array(audio_segments)[kkk]
            #print(kkk, len(single_data) )
            librosaspectrogram = librosa.stft(single_data, n_fft=self.window_size, hop_length=self.hop_size,
                                              win_length=self.window_size, window='hann', center=True,
                                              pad_mode='reflect') ** 2

            melfb = librosa.filters.mel(sr=self.sample_rate, n_fft=self.window_size, n_mels=self.mel_bins, fmin=self.fmin,
                                        fmax=self.fmax).T

            ggg = np.sqrt(librosaspectrogram.real ** 2 + librosaspectrogram.imag ** 2).T

            kkkkk = np.matmul(ggg, melfb)

            final = (np.log10(np.clip(kkkkk, a_min=1e-10, a_max=np.inf)) * 10) - 10.0 * np.log10(np.maximum(1e-10, 1.0))

            #print(kkk, len(final) )
            all_audio_segments.append(final)
        # audio_segments = torch.unsqueeze(torch.tensor(audio_segments), 0)
        audio_segments = move_data_to_device(np.array(all_audio_segments), self.device)
        time_preprocessing = self.__current_time() - time_preprocessing_start
        time_forward_start = self.__current_time()
        self.model.eval()
        #import pdb; pdb.set_trace()
        with torch.no_grad():
            #if len(audio_segments)>2:
            #    import pdb; pdb.set_trace()
            print(audio_segments.shape)
            clip_output = self.model(audio_segments)[0].reshape(
                n_clips, n_segs, self.classes_num)
            #print(clip_output.shape)
            clip_output_dict = {'clipwise_output': torch.max(
                np.exp(clip_output.data.cpu()), axis=1).values}
            #clip_output_dict = {'clipwise_output': torch.mean(np.exp(clip_output.data.cpu()), axis = 1)}
        #print(np.exp(clip_output.data.cpu()))
        time_forward = self.__current_time() - time_forward_start

        time_postprocessing_start = self.__current_time()
        #import pdb; pdb.set_trace()
        for clip_idx in range(clip_output_dict['clipwise_output'].shape[0]):
            clipwise_output = clip_output_dict['clipwise_output'][clip_idx]
            start_time = float(clip_idx * self.audio_clip_length)
            end_time = min(start_time + self.audio_clip_length, audio_length)
            if isinstance(data, dict):
                for i in range(self.classes_num):
                    label = self.names.labels[i]
                    
                    if clipwise_output[i] >= label['threshold']:
                        prediction_dict[audio_file].append({'class': label['name'], 'score': clipwise_output[i].item(),
                                                            'id': label['id'], 'display_name': label['name'], 'start_time': start_time,
                                                            'end_time': end_time})
                predictions = prediction_dict
            else:
                for i in range(self.classes_num):
                    label = self.names.labels[i]
                    if clipwise_output[i] >= label['threshold']:
                        #print(label)
                        prediction_list.append({'class': label['name'], 'score': clipwise_output[i].item(),
                                                'id': label['id'], 'display_name': label['name'], 'start_time': start_time,
                                                'end_time': end_time})
                predictions = prediction_list

        time_postprocessing = self.__current_time() - time_postprocessing_start
        time_total = self.__current_time() - time_total_start
        timers = {"DetectTotalTime": time_total,
                  "DetectPreprocessingTime": time_preprocessing,
                  "DetectPostprocessingTime": time_postprocessing,
                  "DetectForwardTime": time_forward}
        if self.debug_mode == True:
            probs = torch.max(
                clip_output_dict['clipwise_output'], axis=0).values.numpy().tolist()
            return dict(predictions=predictions, timers=timers, probs=probs)
        else:
            return dict(predictions=predictions, timers=timers)

    def val_audio_ingenic(self, data):
        self.model.eval()
        with torch.no_grad():
            #if len(audio_segments)>2:
            #    import pdb; pdb.set_trace()
            clip_output = self.model(data)[0].reshape(
                data.shape[0], 1, self.classes_num)
            #import pdb; pdb.set_trace()
            #print(clip_output.shape)
            clip_output_dict = {'clipwise_output': torch.max(
                np.exp(clip_output.data.cpu()), axis=1).values}

        #import pdb; pdb.set_trace()
        time_postprocessing_start = self.__current_time()
        prediction_list = []

        for clip_idx in range(clip_output_dict['clipwise_output'].shape[0]):
            audio_length = 2
            clipwise_output = clip_output_dict['clipwise_output'][clip_idx]
            start_time = float(clip_idx * self.audio_clip_length)
            end_time = min(start_time + self.audio_clip_length, audio_length)

            for i in range(self.classes_num):
                label = self.names.labels[i]
                if clipwise_output[i] >= label['threshold']:
                    #print(label)
                    prediction_list.append({'class': label['name'], 'score': clipwise_output[i].item(),
                                            'id': label['id'], 'display_name': label['name'], 'start_time': start_time,
                                            'end_time': end_time})
            predictions = prediction_list

        time_postprocessing = self.__current_time() - time_postprocessing_start
        time_total = self.__current_time() - 0
        timers = {"DetectTotalTime": time_total,
                  "DetectPreprocessingTime": 0,
                  "DetectPostprocessingTime": time_postprocessing,
                  "DetectForwardTime": 0}
        #import pdb; pdb.set_trace()
        probs = clip_output_dict['clipwise_output'].numpy()
        #print(probs)
        return dict(predictions=predictions, timers=timers, probs = probs)


    def classify_audio_files(self, data):
       
        time_forward_start = self.__current_time()
        self.model.eval()
        #import pdb; pdb.set_trace()
        with torch.no_grad():
            #if len(audio_segments)>2:
            #    import pdb; pdb.set_trace()
            clip_output = self.model(audio_segments)[0].reshape(
                n_clips, n_segs, self.classes_num)
            #print(clip_output.shape)
            clip_output_dict = {'clipwise_output': torch.max(
                np.exp(clip_output.data.cpu()), axis=1).values}
            #clip_output_dict = {'clipwise_output': torch.mean(np.exp(clip_output.data.cpu()), axis = 1)}
        #print(np.exp(clip_output.data.cpu()))
        time_forward = self.__current_time() - time_forward_start

        time_postprocessing_start = self.__current_time()
        for clip_idx in range(clip_output_dict['clipwise_output'].shape[0]):
            clipwise_output = clip_output_dict['clipwise_output'][clip_idx]
            start_time = float(clip_idx * self.audio_clip_length)
            end_time = min(start_time + self.audio_clip_length, audio_length)
            if isinstance(data, dict):
                for i in range(self.classes_num):
                    label = self.names.labels[i]
                    
                    if clipwise_output[i] >= label['threshold']:
                        prediction_dict[audio_file].append({'class': label['name'], 'score': clipwise_output[i].item(),
                                                            'id': label['id'], 'display_name': label['name'], 'start_time': start_time,
                                                            'end_time': end_time})
                predictions = prediction_dict
            else:
                for i in range(self.classes_num):
                    label = self.names.labels[i]
                    if clipwise_output[i] >= label['threshold']:
                        #print(label)
                        prediction_list.append({'class': label['name'], 'score': clipwise_output[i].item(),
                                                'id': label['id'], 'display_name': label['name'], 'start_time': start_time,
                                                'end_time': end_time})
                predictions = prediction_list

        time_postprocessing = self.__current_time() - time_postprocessing_start
        time_total = self.__current_time() - time_total_start
        timers = {"DetectTotalTime": time_total,
                  "DetectPreprocessingTime": time_preprocessing,
                  "DetectPostprocessingTime": time_postprocessing,
                  "DetectForwardTime": time_forward}
        if self.debug_mode == True:
            probs = torch.max(
                clip_output_dict['clipwise_output'], axis=0).values.numpy().tolist()
            return dict(predictions=predictions, timers=timers, probs=probs)
        else:
            return dict(predictions=predictions, timers=timers)


    def classify_audio(self, data, orig_sample_rate=8000):
        '''
        classify a waveform type audio
        data: dict of {key: waveform}, or waveform.
              waveform (array): darray of sound amplitudes (audio_clip_length, )
        '''
        time_total_start = time_preprocessing_start = self.__current_time()
        if isinstance(data, dict):
            audio_file = list(data.keys())[0]
            waveform = data[audio_file]
            prediction_dict = {audio_file: list()}
        else:
            prediction_list = []
            waveform = data
        if orig_sample_rate != self.target_sample_rate:
            gcd = np.gcd(orig_sample_rate, self.target_sample_rate)
            # there are many other ways to do resampling
            waveform = scipy.signal.resample_poly(
                waveform, self.target_sample_rate // gcd, orig_sample_rate // gcd, axis=-1)

        audio_length = len(waveform)/self.target_sample_rate
        waveform = torch.tensor(waveform[None, :])
        if waveform.shape[1] < self.sample_length:
            n_clips = 1
            n_segs = int(ceil(
                waveform.shape[1] / ((self.segment_length - self.overlap_length) * self.target_sample_rate)))
            audio_segments = pad_split_input(
                waveform, n_clips, n_segs, self.segment_length, self.overlap_length, self.target_sample_rate)
        else:
            n_clips = int(ceil(waveform.shape[1] / self.sample_length))
            audio_clips = audio_sample_pad(
                waveform, n_clips, self.sample_length)
            n_segs = int(ceil(self.sample_length / ((self.segment_length -
                         self.overlap_length) * self.target_sample_rate)))
            audio_segments = pad_split_input(
                audio_clips, n_clips, n_segs, self.segment_length, self.overlap_length, self.target_sample_rate)
        audio_segments = move_data_to_device(audio_segments, self.device)
        time_preprocessing = self.__current_time() - time_preprocessing_start
        time_forward_start = self.__current_time()
        self.model.eval()
        with torch.no_grad():
            clip_output = self.model(audio_segments)[0].reshape(
                n_clips, n_segs, self.classes_num)
            clip_output_dict = {'clipwise_output': torch.max(
                np.exp(clip_output.data.cpu()), axis=1).values}
            #clip_output_dict = {'clipwise_output': torch.mean(np.exp(clip_output.data.cpu()), axis = 1)}
        #import pdb; pdb.set_trace()
        time_forward = self.__current_time() - time_forward_start

        time_postprocessing_start = self.__current_time()
        for clip_idx in range(clip_output_dict['clipwise_output'].shape[0]):
            clipwise_output = clip_output_dict['clipwise_output'][clip_idx]
            start_time = float(clip_idx * self.audio_clip_length)
            end_time = min(start_time + self.audio_clip_length, audio_length)
            if isinstance(data, dict):
                for i in range(self.classes_num):
                    label = self.names.labels[i]
                    if clipwise_output[i] >= label['threshold']:
                        prediction_dict[audio_file].append({'class': label['name'], 'score': clipwise_output[i].item(),
                                                            'id': label['tag_id'], 'display_name': label['display_name'], 'start_time': start_time,
                                                            'end_time': end_time})
                predictions = prediction_dict
            else:
                for i in range(self.classes_num):
                    label = self.names.labels[i]
                    if clipwise_output[i] >= label['threshold']:
                        prediction_list.append({'class': label['name'], 'score': clipwise_output[i].item(),
                                                'id': label['tag_id'], 'display_name': label['display_name'], 'start_time': start_time,
                                                'end_time': end_time})
                predictions = prediction_list

        time_postprocessing = self.__current_time() - time_postprocessing_start
        time_total = self.__current_time() - time_total_start
        timers = {"DetectTotalTime": time_total,
                  "DetectPreprocessingTime": time_preprocessing,
                  "DetectPostprocessingTime": time_postprocessing,
                  "DetectForwardTime": time_forward}
        if self.debug_mode == True:
            probs = torch.max(
                clip_output_dict['clipwise_output'], axis=0).values.numpy().tolist()
            return dict(predictions=predictions, timers=timers, probs=probs)
        else:
            return dict(predictions=predictions, timers=timers)
