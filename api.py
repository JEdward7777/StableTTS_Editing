import torch
import torch.nn as nn

from dataclasses import asdict

from utils.audio import LogMelSpectrogram
from config import ModelConfig, MelConfig
from models.model import StableTTS

from text import symbols
from text import cleaned_text_to_sequence
from text.mandarin import chinese_to_cnm3
from text.english import english_to_ipa2
from text.japanese import japanese_to_ipa2


from datas.dataset import intersperse
from utils.audio import load_and_resample_audio

import josh_hacking

def get_vocoder(model_path, model_name='ffgan') -> nn.Module:
    if model_name == 'ffgan':
        # training or changing ffgan config is not supported in this repo
        # you can train your own model at https://github.com/fishaudio/vocoder
        from vocoders.ffgan.model import FireflyGANBaseWrapper
        vocoder = FireflyGANBaseWrapper(model_path)
        
    elif model_name == 'vocos':
        from vocoders.vocos.models.model import Vocos
        from config import VocosConfig, MelConfig
        vocoder = Vocos(VocosConfig(), MelConfig())
        vocoder.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
        vocoder.eval()
        
    else:
        raise NotImplementedError(f"Unsupported model: {model_name}")
        
    return vocoder

class StableTTSAPI(nn.Module):
    def __init__(self, tts_model_path, vocoder_model_path, vocoder_name='ffgan'):
        super().__init__()

        self.mel_config = MelConfig()
        self.tts_model_config = ModelConfig()
        
        self.mel_extractor = LogMelSpectrogram(**asdict(self.mel_config))
        
        # text to mel spectrogram
        self.tts_model = StableTTS(len(symbols), self.mel_config.n_mels, **asdict(self.tts_model_config))
        self.tts_model.load_state_dict(torch.load(tts_model_path, map_location='cpu', weights_only=True))
        self.tts_model.eval()
        
        # mel spectrogram to waveform
        self.vocoder_model = get_vocoder(vocoder_model_path, vocoder_name)
        self.vocoder_model.eval()
        
        self.g2p_mapping = {
            'chinese': chinese_to_cnm3,
            'japanese': japanese_to_ipa2,
            'english': english_to_ipa2,
        }
        self.supported_languages = self.g2p_mapping.keys()
        
    @ torch.inference_mode()
    def inference(self, text, ref_audio, language, step, temperature=1.0, length_scale=1.0, solver=None, cfg=3.0, prefix=None, postfix=None,
                  prefix_text=None, suffix_text=None):
        device = next(self.parameters()).device
        phonemizer = self.g2p_mapping.get(language)
        
        text = phonemizer(text)
        text = torch.tensor(intersperse(cleaned_text_to_sequence(text), item=0), dtype=torch.long, device=device).unsqueeze(0)
        text_length = torch.tensor([text.size(-1)], dtype=torch.long, device=device)
        
        ref_audio = load_and_resample_audio(ref_audio, self.mel_config.sample_rate).to(device)
        ref_audio = self.mel_extractor(ref_audio)

        if prefix is not None:
            prefix = load_and_resample_audio(prefix, self.mel_config.sample_rate).to(device)
            prefix = self.mel_extractor(prefix)
        
        if postfix is not None:
            postfix = load_and_resample_audio(postfix, self.mel_config.sample_rate).to(device)
            postfix = self.mel_extractor(postfix)

        if prefix_text is not None:
            prefix_text = phonemizer(prefix_text)
            prefix_text = torch.tensor(intersperse(cleaned_text_to_sequence(prefix_text), item=0), dtype=torch.long, device=device).unsqueeze(0)
        
        if suffix_text is not None:
            suffix_text = phonemizer(suffix_text)
            suffix_text = torch.tensor(intersperse(cleaned_text_to_sequence(suffix_text), item=0), dtype=torch.long, device=device).unsqueeze(0)
        
        mel_output = self.tts_model.synthesise(text, text_length, step, temperature, ref_audio, length_scale, solver, cfg, prefix=prefix, postfix=postfix, prefix_text=prefix_text, suffix_text=suffix_text)['decoder_outputs']

        #josh_hacking.plot_mel_spectrogram(mel_output, self.mel_config)

        audio_output = self.vocoder_model(mel_output)
        return audio_output.cpu(), mel_output.cpu()
    
    def get_params(self):
        tts_param = sum(p.numel() for p in self.tts_model.parameters()) / 1e6
        vocoder_param = sum(p.numel() for p in self.vocoder_model.parameters()) / 1e6
        return tts_param, vocoder_param
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tts_model_path = './checkpoints/checkpoint_0.pt'
    vocoder_model_path = './vocoders/pretrained/vocos.pt'
    
    model = StableTTSAPI(tts_model_path, vocoder_model_path, 'vocos')
    model.to(device)
    
    #text = '樱落满殇祈念集……殇歌花落集思祈……樱花满地集于我心……揲舞纷飞祈愿相随……'
    #audio = './audio_1.wav'
    #language = 'chinese'

    # text = "In the Beginning God created the Heavens and the Earth."
    # audio = "./audios/josh.wav"
    # language = 'english'

    prefix_text = "this I "
    text = " understand, "
    suffix_text = "for the "
    audio = "./ref_full.wav"
    prefix = "./ref_lead_in.wav"
    postfix = "./ref_lead_out.wav"
    language = 'english'

#     text = """

# 3 Now there was a Pharisee, a man named Nicodemus who was a member of the Jewish ruling council. 2 He came to Jesus at night and said, “Rabbi, we know that you are a teacher who has come from God. For no one could perform the signs you are doing if God were not with him.”

# 3 Jesus replied, “Very truly I tell you, no one can see the kingdom of God unless they are born again.[a]”

# 4 “How can someone be born when they are old?” Nicodemus asked. “Surely they cannot enter a second time into their mother’s womb to be born!”

# 5 Jesus answered, “Very truly I tell you, no one can enter the kingdom of God unless they are born of water and the Spirit. 6 Flesh gives birth to flesh, but the Spirit[b] gives birth to spirit. 7 You should not be surprised at my saying, ‘You[c] must be born again.’ 8 The wind blows wherever it pleases. You hear its sound, but you cannot tell where it comes from or where it is going. So it is with everyone born of the Spirit.”[d]
# """
    
    audio_output, mel_output = model.inference(text, audio, language, 10, solver='dopri5', cfg=3, prefix=prefix, postfix=postfix, prefix_text=prefix_text, suffix_text=suffix_text)

    print(audio_output.shape)
    print(mel_output.shape)
    
    import torchaudio
    torchaudio.save('output.wav', audio_output, MelConfig().sample_rate)
    
    
