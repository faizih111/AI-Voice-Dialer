
import base64
from google import genai

client = genai.Client(
    http_options={"api_version": "v1beta"}
)

print("Generating speech...")

try:
    interaction = client.interactions.create(
        model="gemini-3.8-flash-tts",
        input=[{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": "Hello! Welcome to our AI voice dialer.",
                "annotations": [{
                    "type": "speech_metadata",
                    "style": "friendly and professional"
                }]
            }]
        }],
        response_format={"type": "audio"},
        generation_config={
            "speech_config": [
                {"voice": "Kore"}
            ]
        }
    )

    audio_data = base64.b64decode(
        interaction.output_audio.data
    )

    with open("test_voice.wav", "wb") as file:
        file.write(audio_data)

    print("SUCCESS! Audio saved as test_voice.wav")

except Exception as error:
    print("TTS ERROR:", repr(error))

