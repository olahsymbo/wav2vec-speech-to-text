import torch


class Preprocessor:
    def __init__(self):
        self.target_sample_rate = 16000 
    
    def resample(self, waveform):
        resampled_waveform = (
            waveform[:, :self.target_sample_rate] 
            if waveform.shape[1] > self.target_sample_rate 
            else torch.nn.functional.pad(
                waveform, 
                (0, self.target_sample_rate - waveform.shape[1])
            )
        )
        return resampled_waveform
        
    def normalize_tensor(self, mel):
        mean = mel.mean()
        std = mel.std()
        return (mel - mean) / (std + 1e-6)