from pydub import AudioSegment

def split_audio(audio_path):

    audio = AudioSegment.from_file(audio_path)

    chunk_length = 30000

    chunks = []

    for i in range(0, len(audio), chunk_length):

        chunk = audio[i:i+chunk_length]

        filename = f"uploads/chunk_{i}.wav"

        chunk.export(filename, format="wav")

        chunks.append(filename)

    return chunks