# ------------------ streamlit first ------------------
import streamlit as st
st.set_page_config(page_title="Cervical Spine Fracture Detection", layout="wide")

# ------------------ imports ------------------
import torch
import yaml
import timm
from PIL import Image
from io import BytesIO
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
import os

# ------------------ Load Config ------------------
with open("configs.yaml") as f:
    cfg = yaml.safe_load(f)

device = 'cuda' if torch.cuda.is_available() and cfg['device'] in ['auto','cuda'] else 'cpu'

# ------------------ Model Loading ------------------
@st.cache_resource
def load_model(checkpoint_path, cfg):
    if not os.path.exists(checkpoint_path):
        st.error("Checkpoint not found! Run training first to generate 'best.pth'")
        st.stop()
    model = timm.create_model(cfg['backbone'], pretrained=True, num_classes=cfg['num_classes'])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

checkpoint_path = "outputs/checkpoints/best.pth"
model = load_model(checkpoint_path, cfg)

# ------------------ Image Preprocessing ------------------
def preprocess_image(img: Image.Image, img_size=224):
    img = img.resize((img_size, img_size))
    img_tensor = torch.tensor(
        (torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
         .float()
         .view(img.size[1], img.size[0], 3)
         / 255.0)
    ).permute(2,0,1)
    # Normalize
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.unsqueeze(0).to(device)

# ------------------ LLM Suggestions ------------------
def get_llm_suggestion(pred_class, conf, dataset_context, groq_api_key):
    client = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")
    prompt = f"""
Model Prediction: {'Fracture' if pred_class==1 else 'Normal'}
Confidence: {conf:.4f}
Dataset context: {dataset_context}

Analyze this prediction in detail and provide:
1) A detailed explaination about this is C1-C7 spine disorders if its fracture
2) Give the Prescription of the image (modality, view, anatomy, etc.)
3) Potential issues or uncertainties in the image (artifacts, low contrast, poor angle, etc.)
4) Suggestions to how we can help for the fracture of the outcome if its normal how we can prevent from that suggest.
5) guidances of the fracture if its normal how to be better.
give treatment advice. Provide in a clear, professional, research-level style.
"""
    messages = [
        SystemMessage(content="You are an expert radiology research assistant."),
        HumanMessage(content=prompt)
    ]
    response = client(messages)
    return response.content

# ------------------ Streamlit UI ------------------
st.title("🩻 Cervical Spine Fracture Detection & LLM Analysis")

with st.sidebar:
    st.header("Settings")
    dataset_context = st.text_area("Dataset Context", "Images from Roboflow dataset ~4115, classes: fracture, normal")
    groq_api_key = st.text_input("Groq API Key", type="password")

uploaded_file = st.file_uploader("Upload CT Image", type=["png","jpg","jpeg"])

if uploaded_file and groq_api_key:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1,1])
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Prediction
    tensor = preprocess_image(img, cfg['img_size'])
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        pred_class = int(probs.argmax())
        conf = float(probs[pred_class])
    
    with col2:
        st.subheader("Model Prediction")
        st.metric(label="Predicted Class", value="Fracture" if pred_class==1 else "Normal")
        st.metric(label="Confidence", value=f"{conf:.4f}")
    
    # LLM Suggestion
    with st.spinner("Generating advanced LLM suggestions..."):
        llm_suggestion = get_llm_suggestion(pred_class, conf, dataset_context, groq_api_key)
    
    st.subheader("📌 LLM Analytical Suggestions")
    st.write(llm_suggestion)

    # Download results
    st.download_button(
        label="Download LLM Suggestions",
        data=llm_suggestion,
        file_name="llm_suggestions.txt",
        mime="text/plain"
    )
