from pydub import AudioSegment

def merge_audio(files):

    final = AudioSegment.empty()

    for f in files:

        audio = AudioSegment.from_file(f)

        final += audio

    output_path = "outputs/final.wav"

    final.export(output_path, format="wav")

    return output_path