import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

@st.cache_resource
def load_model():
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        device=0,
        pipeline_kwargs={
            "max_new_tokens": 200,
            "temperature": 0.5
        }
    )

    return ChatHuggingFace(llm=llm)

model = load_model()

template = PromptTemplate(
    template="""
You are an expert AI researcher and educator.

Explain the research paper: {paper_input}

Explanation Style: {style_input}

Explanation Length: {length_input}

Additional User Instructions:
{additional_prompt}

Provide a clear and accurate explanation.
""",
    input_variables=[
        "paper_input",
        "style_input",
        "length_input",
        "additional_prompt"
    ]
)

st.title("Research Paper Explainer")

paper_input = st.selectbox(
    "Select Research Paper",
    [
        "Attention Is All You Need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented",
        "Mathematical"
    ]
)

length_input = st.selectbox(
    "Explanation Length",
    [
        "Short (1-2 paragraphs)",
        "Medium (3-5 paragraphs)",
        "Long (Detailed Explanation)"
    ]
)

additional_prompt = st.text_area(
    "Additional Instructions (Optional)",
    placeholder="Example: Use real-world examples and simple language."
)

if st.button("Summarize"):

    chain = template | model

    result = chain.invoke({
        "paper_input": paper_input,
        "style_input": style_input,
        "length_input": length_input,
        "additional_prompt": additional_prompt
    })

    st.write(result.content)