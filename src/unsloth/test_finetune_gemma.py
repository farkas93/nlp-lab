import streamlit as st
import torch
from unsloth import FastModel
import pandas as pd
import os
import time
import csv
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="LLM Comparison Tool",
    layout="wide",
)

# Constants
max_seq_length = 1024
OUT_MODEL_NAME = "outputs"
CSV_FILE = "model_comparison_results.csv"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Default test problems
DEFAULT_PROMPTS = [
    "If John has 5 apples and gives 2 to Mary, then buys 3 more, how many apples does John have now?",
    "What is 25 * 13?",
    "A train travels at 60 mph for 3 hours, then at 80 mph for 2 hours. How far did it travel in total?"
]

@st.cache_resource
def load_ft_model():
    """Load fine-tuned model and tokenizer"""
    st.info("Loading fine-tuned model...")
    
    try:
        ft_model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/gemma-3-1b-it",
            max_seq_length=max_seq_length,
            device_map="auto",
        )
        
        # Apply the trained LoRA weights
        ft_model = FastModel.get_peft_model(
            ft_model,
            model_path=OUT_MODEL_NAME,
        )
        
        ft_model.eval()
        return ft_model, tokenizer
    except Exception as e:
        st.error(f"Error loading fine-tuned model: {str(e)}")
        st.stop()

@st.cache_resource
def load_vanilla_model():
    """Load vanilla model"""
    st.info("Loading vanilla model...")
    
    try:
        vanilla_model, tokenizer = FastModel.from_pretrained(
            model_name="unsloth/gemma-3-1b-it",
            max_seq_length=max_seq_length,
            device_map="auto",
        )
        
        vanilla_model.eval()
        return vanilla_model, tokenizer
    except Exception as e:
        st.error(f"Error loading vanilla model: {str(e)}")
        st.stop()

def generate_response(model, tokenizer, prompt):
    """Generate a response from the model"""
    try:
        # Convert prompt to model inputs
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to appropriate device if model is not quantized
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
        end_time = time.time()
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generation_time = end_time - start_time
        
        return response, generation_time
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return f"Error: {str(e)}", 0.0

def load_history():
    """Load comparison history from CSV"""
    if os.path.exists(CSV_FILE):
        return pd.read_csv(CSV_FILE)
    return pd.DataFrame(columns=["timestamp", "prompt", "fine_tuned_response", "vanilla_response", "ft_time", "vanilla_time"])

def save_to_csv(timestamp, prompt, ft_response, vanilla_response, ft_time, vanilla_time):
    """Save comparison results to CSV"""
    file_exists = os.path.isfile(CSV_FILE)
    
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["timestamp", "prompt", "fine_tuned_response", "vanilla_response", "ft_time", "vanilla_time"])
        writer.writerow([timestamp, prompt, ft_response, vanilla_response, ft_time, vanilla_time])

# App title
st.title("Fine-tuned vs Vanilla Model Comparison")
st.write(f"Running on device: {device}")

# Load past comparisons
history = load_history()

# Input interface for new prompts
st.subheader("Create a new comparison")
with st.form("prompt_form"):
    new_prompt = st.text_area("Enter a prompt:", height=100)
    submitted = st.form_submit_button("Generate Responses")

# Handle form submission
if submitted and new_prompt:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # First load the fine-tuned model and generate response
    with st.status("Processing with fine-tuned model...") as status:
        status.update(label="Loading fine-tuned model...")
        ft_model, tokenizer = load_ft_model()
        
        status.update(label="Generating response with fine-tuned model...")
        ft_response, ft_time = generate_response(ft_model, tokenizer, new_prompt)
        status.update(label="Fine-tuned model complete!", state="complete")
    
    # Then load the vanilla model and generate response
    with st.status("Processing with vanilla model...") as status:
        status.update(label="Loading vanilla model...")
        vanilla_model, _ = load_vanilla_model()
        
        status.update(label="Generating response with vanilla model...")
        vanilla_response, vanilla_time = generate_response(vanilla_model, tokenizer, new_prompt)
        status.update(label="Vanilla model complete!", state="complete")
    
    # Save to CSV
    save_to_csv(timestamp, new_prompt, ft_response, vanilla_response, ft_time, vanilla_time)
    
    # Reload history
    history = load_history()
    
    st.success("Responses generated and saved!")

# Run default prompts if history is empty
if history.empty and not submitted:
    st.info("No previous comparisons found. Running default prompts...")
    
    for prompt in DEFAULT_PROMPTS:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with st.status(f"Processing prompt: {prompt[:30]}...") as status:
            # Load and run fine-tuned model first
            status.update(label="Loading fine-tuned model...")
            ft_model, tokenizer = load_ft_model()
            
            status.update(label="Generating with fine-tuned model...")
            ft_response, ft_time = generate_response(ft_model, tokenizer, prompt)
            
            # Then load and run vanilla model
            status.update(label="Loading vanilla model...")
            vanilla_model, _ = load_vanilla_model()
            
            status.update(label="Generating with vanilla model...")
            vanilla_response, vanilla_time = generate_response(vanilla_model, tokenizer, prompt)
            
            status.update(label="Completed!", state="complete")
        
        # Save to CSV
        save_to_csv(timestamp, prompt, ft_response, vanilla_response, ft_time, vanilla_time)
    
    # Reload history
    history = load_history()
    st.success("Default prompts processed!")

# Display history
if not history.empty:
    st.subheader("Comparison Results")
    
    # Show most recent comparison first
    history = history.sort_values("timestamp", ascending=False)
    
    for index, row in history.iterrows():
        st.write("---")
        st.write(f"**Time:** {row['timestamp']}")
        
        # Create three columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Prompt")
            st.text_area("", row["prompt"], height=200, key=f"prompt_{index}", disabled=True)
        
        with col2:
            st.markdown("### Fine-tuned Model")
            st.text_area("", row["fine_tuned_response"], height=400, key=f"ft_{index}", disabled=True)
            
            # Check for training indicators
            has_reasoning_tags = "<start_working_out>" in row["fine_tuned_response"] and "<end_working_out>" in row["fine_tuned_response"]
            has_solution_tags = "<SOLUTION>" in row["fine_tuned_response"] and "</SOLUTION>" in row["fine_tuned_response"]
            
            if has_reasoning_tags and has_solution_tags:
                st.success("✓ Following trained format")
            elif has_reasoning_tags or has_solution_tags:
                st.warning("~ Partially following trained format")
            else:
                st.error("✗ Not following trained format")
                
            st.caption(f"Generation time: {row['ft_time']:.2f} seconds")
        
        with col3:
            st.markdown("### Vanilla Model")
            st.text_area("", row["vanilla_response"], height=400, key=f"vanilla_{index}", disabled=True)
            st.caption(f"Generation time: {row['vanilla_time']:.2f} seconds")

# Add option to clear history
if not history.empty:
    if st.button("Clear History"):
        if os.path.exists(CSV_FILE):
            os.remove(CSV_FILE)
            st.success("History cleared!")
            st.experimental_rerun()