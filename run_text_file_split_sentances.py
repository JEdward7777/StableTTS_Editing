import torchaudio
import torch
import numpy as np
import os

from api import StableTTSAPI
import inflect

def write_out_numbers( text ):
    output = []
    number = []
    p = inflect.engine()
    for char in text:

        if char.isdigit():
            number.append(char)
        else:
            if char == "-":
                char = " - "
            elif char == "—":
                char = ", "
            if len(number) > 0:
                word_text = p.number_to_words( int(''.join(number)) )
                output.append( word_text )
                number = []
            output.append( char )
    if len(number) > 0:
        word_text = p.number_to_words( int(''.join(number)) )
        output.append( word_text )
    return ''.join(output)

def main():
    tts_model_path = './checkpoints/checkpoint_2850.pt'
    text_file_path = '/home/lansford/bible_token_models/obs_data/chunking_obs_for_stable_tts/data/en_obs_v8.txt'
    output_folder = '/home/lansford/bible_token_models/obs_data/chunking_obs_for_stable_tts/data/tts_output_split'
    vocoder_model_path = './vocoders/pretrained/firefly-gan-base-generator.ckpt'
    reference_audio_file = '/home/lansford/bible_token_models/obs_data/chunking_obs_for_stable_tts/data/en_obs_v6_mp3_128kbps/en_obs_02_128kbps.mp3'

    language = 'english'
    step = 25
    temperature = 1
    length_scale = 1
    solver = 'dopri5'
    cfg = 3
    vocoder_type = 'ffgan'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'

    
    model = StableTTSAPI(tts_model_path, vocoder_model_path, vocoder_type).to(device)


    with open( text_file_path, 'r', encoding='utf-8') as f:
        for count, text in enumerate(f):
            text = text.strip()

            text_collection = [text]


            #split the text into sentences.
            splitters_for_sentances = ['. ', '? ', '! ']
            for splitter in splitters_for_sentances:
                new_collection = []
                for text_piece in text_collection:
                    if splitter in text_piece:
                        for sentence in text_piece.split(splitter):
                            if not text_piece.endswith(sentence):
                                sentence += splitter
                            new_collection.append(sentence)
                    else:
                        new_collection.append(text_piece)
                text_collection = new_collection

            #merge short fragments.
            new_collection = []
            build = ""
            for text_piece in text_collection:
                build += text_piece
                if len(build) > 40:
                    new_collection.append(build)
                    build = ""
            if len(build) > 0:
                new_collection.append(build)
            text_collection = new_collection



            combined_audio = None

            for sentence in text_collection:
                sentence = write_out_numbers(sentence)
                print( "Running the sentence:", sentence[:100] )

                audio, _ = model.inference(sentence, reference_audio_file, language, step, temperature, length_scale, solver, cfg )

                max_val = torch.max(torch.abs(audio))
                if max_val > 1:
                    audio = audio / max_val
                
                audio_output = (model.mel_config.sample_rate, (audio.cpu().squeeze(0).numpy() * 32767).astype(np.int16)) # (samplerate, int16 audio) for gr.Audio

                if combined_audio is None:
                    combined_audio = audio_output
                else:
                    combined_audio = (combined_audio[0], np.concatenate( (combined_audio[1], audio_output[1]), axis=0 ) )


            #create folder if it doesn't exist:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            audio_tensor = torch.tensor(combined_audio[1])
            audio_tensor = audio_tensor.unsqueeze(0)

            torchaudio.save(f'{output_folder}/{count+1:04d}.mp3', audio_tensor, combined_audio[0], format='mp3')


    

if __name__ == '__main__':
    main()
