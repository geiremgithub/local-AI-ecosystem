import os
import streamlit as st
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MAPPESTI = r"C:\working\python\dokumenter"
BILDE_ENDELSER = (".png", ".jpg", ".jpeg")

# Tittel på nettsiden
st.set_page_config(page_title="Tegningsassistent", layout="wide")
st.title("🗺️ Visuell AI - Tegningsassistent")

# Last inn modellen kun én gang (for ytelse)
@st.cache_resource
def last_inn_modell():
    vl_modell = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    vl_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    return vl_modell, vl_processor

st.info("Laster inn AI-modell...")
vl_modell, vl_processor = last_inn_modell()

# Finn bilder i mappen
bilde_filer = [
    os.path.join(MAPPESTI, f) for f in os.listdir(MAPPESTI) 
    if os.path.splitext(f)[1].lower() in BILDE_ENDELSER
]

if bilde_filer:
    valgt_bilde = bilde_filer[0]
    
    # Vis bildet på nettsiden
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Teknisk tegning")
        st.image(valgt_bilde, use_container_width=True)
        
    with col2:
        st.subheader("Spør AI om tegningen")
        sporsmal = st.text_input("Hva lurer du på om tegningen?", "Hvor finner jeg kontrollrommet på Dekk 2?")
        
        if st.button("Analyser tegning", type="primary"):
            with st.spinner("AI analyserer bildet..."):
                image = Image.open(valgt_bilde)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {
                                "type": "text",
                                "text": (
                                    f"Du ser på en teknisk tegning. Besvar følgende spørsmål på norsk basert på det du ser på bildet:\n"
                                    f"Spørsmål: {sporsmal}\n"
                                    f"Søk grundig over HELE tegningen. List opp ALLE rom, plasseringer og detaljer du finner som passer til spørsmålet."
                                ),
                            },
                        ],
                    }
                ]

                text = vl_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = vl_processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                )
                inputs = inputs.to("cuda" if torch.cuda.is_available() else "cpu")

                generated_ids = vl_modell.generate(**inputs, max_new_tokens=300)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = vl_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                st.success("Analysert!")
                st.write(output_text[0].strip())
else:
    st.error("Ingen bilder/tegninger ble funnet i mappen.")