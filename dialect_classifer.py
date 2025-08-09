import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from peft import PeftModel


st.set_page_config(page_title="Arabic Dialect Classifier", page_icon="🌍")


st.image("logo.png", width=150) 

st.title("🌍 Arabic Dialect Classifier")


@st.cache_resource
def load_pipeline():
    base_model_name = "UBC-NLP/MARBERTv2"     
    lora_model_name = "Zakaria279/MARBERTv2-lora"  # LoRA adapter

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=4)

 
    model = PeftModel.from_pretrained(base_model, lora_model_name)

    return pipeline("text-classification", model=model, tokenizer=tokenizer)


user_input = st.text_area("✍️ اكتب جملة بالعربية:", "")

# تحميل الموديل عند الضغط على الزر فقط
if st.button("🔍 صنّف اللهجة"):
    if user_input.strip():
        with st.spinner("⏳ جارٍ تحميل النموذج وتصنيف النص..."):
            classifier = load_pipeline()
            results = classifier(user_input, truncation=True)
        
        id2label = {0: "مصري", 1: "خليجي", 2: "شامي", 3: "شمال افريقيا"}
        label = results[0]['label']
        score = results[0]['score']

        
        label_index = int(label.replace("LABEL_", ""))
        st.success(f"**اللهجة:** {id2label[label_index]}")
        st.write(f"**نسبة الثقة:** {score:.2%}")
    else:
        st.warning("⚠️ الرجاء إدخال نص أولاً.")







