from kokoro import KPipeline

# from IPython.display import display, Audio # Removed as not needed in script
import soundfile as sf
import torch

pipeline = KPipeline(lang_code="a")  # 'a' for American English
text = """
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
"""
generator = pipeline(text, voice="af_heart")

print("Generating audio...")
for i, (gs, ps, audio) in enumerate(generator):
    print(f"Segment {i}: Graphemes: {gs} | Phonemes: {ps}")
    # display(Audio(data=audio, rate=24000, autoplay=i==0)) # IPython display won't work directly in script
    output_filename = f"audio_{i}.wav"
    sf.write(output_filename, audio, 24000)
    print(f"Saved segment {i} to {output_filename}")

print("Audio generation complete.")
