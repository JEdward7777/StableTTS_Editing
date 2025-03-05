import os
os.environ['TMPDIR'] = './temps' # avoid the system default temp folder not having access permissions
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # use huggingfacae mirror for users that could not login to huggingface

import re
import numpy as np
import matplotlib.pyplot as plt
import tempfile

import torch
import gradio as gr

from api import StableTTSAPI

import whisper_timestamped as whisper

from pydub import AudioSegment

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tts_model_path = './checkpoints/checkpoint_0.pt'
vocoder_model_path = './vocoders/pretrained/firefly-gan-base-generator.ckpt'
vocoder_type = 'ffgan'
wisper_model = "tiny.en"

model = StableTTSAPI(tts_model_path, vocoder_model_path, vocoder_type).to(device)


def cut_audio(file_path, start_time, end_time, output_path, format="mp3"):
    # Load audio file
    audio = AudioSegment.from_mp3(file_path)

    # Convert time to milliseconds
    if start_time is not None:
        start_time = start_time * 1000
    if end_time is not None:
        end_time = end_time * 1000

    # Extract segment
    if start_time is not None and end_time is not None:
        audio_segment = audio[start_time:end_time]
    elif start_time is not None:
        audio_segment = audio[start_time:]
    elif end_time is not None:
        audio_segment = audio[:end_time]
    else:
        audio_segment = audio

    # Export segment
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    audio_segment.export(output_path, format=format)

def run_wisper( audio ):
    wisper_audio = whisper.load_audio(audio)
    wisper_model_loaded = whisper.load_model(wisper_model)
    transcription = wisper_model_loaded.transcribe(wisper_audio, word_timestamps=True)
    return transcription

def transcribe( audio ):
    return run_wisper(audio)['text']


def determine_prefix_text( text, wisper_output ):
    prefix_text = ""
    for segment in wisper_output['segments']:
        for word in segment['words']:
            test_prefix_text = prefix_text + word['word']
            if text.startswith(test_prefix_text):
                prefix_text = test_prefix_text

            else:
                return prefix_text, word['start']

    return prefix_text, wisper_output['segments'][-1]['end']

def determine_postfix_text( text, wisper_output ):
    postfix_text = ""
    for segment in reversed(wisper_output['segments']):
        for word in reversed(segment['words']):
            test_postfix_text = word['word'] + postfix_text
            if text.endswith(test_postfix_text):
                postfix_text = test_postfix_text

            else:
                return postfix_text, word['end']

    return postfix_text, 0

@ torch.inference_mode()
def inference(text, reference_audio, language, step, temperature, length_scale, solver, cfg):
    #text = remove_newlines_after_punctuation(text)

    if language == 'chinese':
        text = text.replace(' ', '')

    wisper_output = run_wisper(reference_audio)

    if text != wisper_output['text']:
        prefix_text, start_timestamp = determine_prefix_text(text, wisper_output)
        postfix_text, end_timestamp = determine_postfix_text(text, wisper_output)

        edited_text = text[len(prefix_text):len(text)-len(postfix_text)]

        #get a temp filename using tempfile for the prefix and postfix audio
        prefix_audio = tempfile.mktemp(suffix='.wav')
        postfix_audio = tempfile.mktemp(suffix='.wav')

        #cut the prefix and postfix audio
        cut_audio(reference_audio, None         , start_timestamp, prefix_audio , format="wav")
        cut_audio(reference_audio, end_timestamp, None           , postfix_audio, format="wav")

        audio, mel = model.inference(edited_text, reference_audio, language, step, temperature, length_scale, solver, cfg, prefix_text=prefix_text, suffix_text=postfix_text, prefix=prefix_audio, postfix=postfix_audio)
        
        max_val = torch.max(torch.abs(audio))
        if max_val > 1:
            audio = audio / max_val
        
        audio_output = (model.mel_config.sample_rate, (audio.cpu().squeeze(0).numpy() * 32767).astype(np.int16)) # (samplerate, int16 audio) for gr.Audio
        mel_output = plot_mel_spectrogram(mel.cpu().squeeze(0).numpy()) # get the plot of mel

    else:
        raise gr.Error('The text hasn\'t been changed')
    
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
    
    #from pathlib import Path
    #examples = list(Path('./audios').rglob('*.wav'))

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


                ref_audio_gr = gr.Audio(
                    label="Reference Audio",
                    type="filepath"
                )

                transcribe_btn = gr.Button( "Transcribe" )


                input_text_gr = gr.Textbox(
                    label="Input Text",
                    info="Put your replacement here",
                    value=example_text,
                )

                transcribe_btn.click( fn=transcribe, inputs=ref_audio_gr, outputs=input_text_gr )

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
                #examples = gr.Examples( [["./ref_lead_in.wav","./ref_lead_out.wav"]], [prefix_audio_gr, postfix_audio_gr])

        tts_button.click(inference, [input_text_gr, ref_audio_gr, language_gr, step_gr, temperature_gr, length_scale_gr, solver_gr, cfg_gr], outputs=[audio_gr, mel_gr])

    demo.queue()  
    demo.launch(debug=True, show_api=True, share=True)


if __name__ == '__main__':
    main()