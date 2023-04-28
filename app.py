import streamlit as st
import os
import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement

ALLOWED_EXTENSIONS = {'wav'}

enhance_model = SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    savedir="pretrained_models/metricgan-plus-voicebank",
)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(file):
    noisy, rate = torchaudio.load(file)
    assert rate == 16000, "sampling rate must be 16000"
    # Add fake batch dimension and relative length tensor
    enhanced = enhance_model.enhance_batch(noisy.unsqueeze(0), lengths=torch.tensor([1.]))
    return noisy[0].numpy(), enhanced[0].cpu().numpy(), rate

def main():
    st.set_page_config(page_title="Speech Enhancement", page_icon="🔊", layout="wide")
    st.title("Speech Enhancement Using SpeechBrain - MetricGan+")

    uploaded_file = st.file_uploader("Upload an audio file", type=ALLOWED_EXTENSIONS)

    if uploaded_file is not None:
        if allowed_file(uploaded_file.name):
            with st.spinner("Processing..."):
                noisy, enhanced, sr = process_file(uploaded_file)
            st.audio(noisy, format='audio/wav', start_time=0, sample_rate=sr, caption="Noisy signal")
            st.audio(enhanced, format='audio/wav', start_time=0, sample_rate=sr, caption="Enhanced signal")
            st.download_button(
                label="Download enhanced signal",
                data=enhanced,
                file_name="enhanced.wav",
                mime="audio/wav"
            )
        else:
            st.warning("Invalid file type. Please upload a WAV file.")

if __name__ == '__main__':
    main()