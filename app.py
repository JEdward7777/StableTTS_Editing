import os
os.environ['TMPDIR'] = './temps' # avoid the system default temp folder not having access permissions
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # use huggingfacae mirror for users that could not login to huggingface

import re
import numpy as np
import matplotlib.pyplot as plt

import torch
import gradio as gr

from api import StableTTSAPI

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tts_model_path = './checkpoints/checkpoint_0.pt'
vocoder_model_path = './vocoders/pretrained/firefly-gan-base-generator.ckpt'
vocoder_type = 'ffgan'

model = StableTTSAPI(tts_model_path, vocoder_model_path, vocoder_type).to(device)

@ torch.inference_mode()
def inference(text, prefix_text, postfix_text, prefix_audio, postfix_audio, language, step, temperature, length_scale, solver, cfg):
    #text = remove_newlines_after_punctuation(text)
    
    if language == 'chinese':
        text = text.replace(' ', '')
        
    audio, mel = model.inference(text, prefix_audio, language, step, temperature, length_scale, solver, cfg, prefix_text=prefix_text, suffix_text=postfix_text, prefix=prefix_audio, postfix=postfix_audio)
    
    max_val = torch.max(torch.abs(audio))
    if max_val > 1:
        audio = audio / max_val
    
    audio_output = (model.mel_config.sample_rate, (audio.cpu().squeeze(0).numpy() * 32767).astype(np.int16)) # (samplerate, int16 audio) for gr.Audio
    mel_output = plot_mel_spectrogram(mel.cpu().squeeze(0).numpy()) # get the plot of mel
    
    return audio_output, mel_output

def plot_mel_spectrogram(mel_spectrogram):
    plt.close() # prevent memory leak
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0) # remove white edges
    return fig

def remove_newlines_after_punctuation(text):
    pattern = r'([，。！？、“”‘’《》【】；：,.!?\'\"<>()\[\]{}])\n'
    return re.sub(pattern, r'\1', text)

def main():
    
    from pathlib import Path
    examples = list(Path('./audios').rglob('*.wav'))

    # gradio wabui, reference: https://huggingface.co/spaces/fishaudio/fish-speech-1
    gui_title = 'StableTTS'
    gui_description = """Next-generation TTS model using flow-matching and DiT, inspired by Stable Diffusion 3."""
    example_prefix_text = "this I"
    example_text = " know, "
    example_postfix_text = " for the Bible"
    
    with gr.Blocks(theme=gr.themes.Base()) as demo:
        demo.load(None, None, js="() => {const params = new URLSearchParams(window.location.search);if (!params.has('__theme')) {params.set('__theme', 'light');window.location.search = params.toString();}}")

        with gr.Row():
            with gr.Column():
                gr.Markdown(f"# {gui_title}")
                gr.Markdown(gui_description)

        with gr.Row():
            with gr.Column():

                prefix_audio_gr = gr.Audio(
                    label="Prefix Audio",
                    type="filepath"
                )

                postfix_audio_gr = gr.Audio(
                    label="Postfix Audio",
                    type="filepath"
                )
             
                # ref_audio_gr = gr.Audio(
                #     label="Reference Audio",
                #     type="filepath"
                # )
                
                input_prefix_gr = gr.Textbox(
                    label="Input Prefix Text",
                    info="Put your prefix text here",
                    value=example_prefix_text,
                )

                input_text_gr = gr.Textbox(
                    label="Input Text",
                    info="Put your replacement here",
                    value=example_text,
                )

                input_postfix_gr = gr.Textbox(
                    label="Input Postfix",
                    info="Put your postfix here",
                    value=example_postfix_text,
                )
                
                language_gr = gr.Dropdown(
                    label='Language',
                    choices=list(model.supported_languages),
                    value = 'english'
                )
                
                step_gr = gr.Slider(
                    label='Step',
                    minimum=1,
                    maximum=100,
                    value=25,
                    step=1
                )
                
                temperature_gr = gr.Slider(
                    label='Temperature',
                    minimum=0,
                    maximum=2,
                    value=1,
                )
                
                length_scale_gr = gr.Slider(
                    label='Length_Scale',
                    minimum=0,
                    maximum=5,
                    value=1,
                )
                
                solver_gr = gr.Dropdown(
                    label='ODE Solver',
                    choices=['euler', 'midpoint', 'dopri5', 'rk4', 'implicit_adams', 'bosh3', 'fehlberg2', 'adaptive_heun'],
                    value = 'dopri5'
                )
                
                cfg_gr = gr.Slider(
                    label='CFG',
                    minimum=0,
                    maximum=10,
                    value=3,
                )
                
            with gr.Column():
                mel_gr = gr.Plot(label="Mel Visual")
                audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
                tts_button = gr.Button("\U0001F3A7 Generate / 合成", elem_id="send-btn", visible=True, variant="primary")
                #examples = gr.Examples(examples, ref_audio_gr)
                examples = gr.Examples( [["./ref_lead_in.wav","./ref_lead_out.wav"]], [prefix_audio_gr, postfix_audio_gr])

        tts_button.click(inference, [input_text_gr, input_prefix_gr, input_postfix_gr, prefix_audio_gr, postfix_audio_gr, language_gr, step_gr, temperature_gr, length_scale_gr, solver_gr, cfg_gr], outputs=[audio_gr, mel_gr])

    demo.queue()  
    demo.launch(debug=True, show_api=True, share=True)


if __name__ == '__main__':
    main()