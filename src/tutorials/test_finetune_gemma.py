import streamlit as st
import torch
import pandas as pd
import os
import json
import time
from datetime import datetime
import glob
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="LLM Comparison Tool",
    layout="wide",
)

# Constants
max_seq_length = 1024
OUTPUTS_DIR = "outputs"
COMP_DATA_DIR = "comp_data"
PROMPTS_FILE = "test_prompts.json"
HF_VANILLA_PREFIX = "unsloth/"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure directories exist
os.makedirs(COMP_DATA_DIR, exist_ok=True)

# Function to determine if a model path is local or HF hub
def is_local_path(model_path):
    """Check if the model path is a local directory"""
    return os.path.exists(model_path) and os.path.isdir(model_path)

# Function to scan for available models
@st.cache_data(ttl=60)  # Cache for 60 seconds to avoid rescanning on every refresh
def scan_models():
    """Scan the outputs directory for vanilla models and their fine-tuned versions"""
    models = {}
    
    # Check if outputs directory exists
    if not os.path.exists(OUTPUTS_DIR):
        return models
    
    # Scan for vanilla models (directories in outputs)
    vanilla_models = [d for d in os.listdir(OUTPUTS_DIR) 
                     if os.path.isdir(os.path.join(OUTPUTS_DIR, d))]
    
    for vanilla_model in vanilla_models:
        vanilla_path = os.path.join(OUTPUTS_DIR, vanilla_model)
        
        # Find fine-tuned models (subdirectories)
        finetunes = [d for d in os.listdir(vanilla_path) 
                    if os.path.isdir(os.path.join(vanilla_path, d))]
        
        models[vanilla_model] = finetunes
    
    return models

@st.cache_resource
def load_vanilla_model(vanilla_model_name):
    """Load vanilla model - supports both local paths and HF hub models"""
    with st.status(f"Loading vanilla model: {vanilla_model_name}", expanded=True) as status:
        try:
            # Determine if it's a local path or HF hub model
            model_path = vanilla_model_name
            # if not is_local_path(vanilla_model_name):
            #    status.update(label=f"Loading model from local path: {vanilla_model_name}") 
            # else:
            status.update(label=f"Loading model from Hugging Face Hub: {HF_VANILLA_PREFIX + vanilla_model_name}")
                
            model, tokenizer = FastModel.from_pretrained(
                model_name= HF_VANILLA_PREFIX + model_path,
                max_seq_length=max_seq_length,
                device_map="auto",
            )
            
            model.eval()
            status.update(label=f"Vanilla model {vanilla_model_name} loaded successfully!", state="complete")
            return model, tokenizer
        except Exception as e:
            status.update(label=f"Error loading vanilla model: {str(e)}", state="error")
            st.error(f"Error loading vanilla model: {str(e)}")
            st.stop()

@st.cache_resource
def load_finetuned_model(vanilla_model_name, finetune_name):
    """Load fine-tuned model - supports both local paths and HF hub models"""
    with st.status(f"Loading fine-tuned model: {finetune_name}", expanded=True) as status:
        try:
            # Determine if vanilla model is a local path or HF hub model
            model_path = vanilla_model_name
            if not is_local_path(vanilla_model_name):
                status.update(label=f"Loading base model from Hugging Face Hub: {vanilla_model_name}")
            else:
                status.update(label=f"Loading base model from local path: {vanilla_model_name}")
            
            # Load base model
            model, tokenizer = FastModel.from_pretrained(
                model_name=model_path,
                max_seq_length=max_seq_length,
                device_map="auto",
            )
            
            # Apply the trained LoRA weights
            status.update(label=f"Applying LoRA adapter from {finetune_name}")
            finetune_path = os.path.join(OUTPUTS_DIR, vanilla_model_name, finetune_name)
            model = FastModel.get_peft_model(
                model,
                model_path=finetune_path,
            )
            
            model.eval()
            status.update(label=f"Fine-tuned model {finetune_name} loaded successfully!", state="complete")
            return model, tokenizer
        except Exception as e:
            status.update(label=f"Error loading fine-tuned model: {str(e)}", state="error")
            st.error(f"Error loading fine-tuned model: {str(e)}")
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

def load_test_prompts():
    """Load test prompts from JSON file"""
    if os.path.exists(PROMPTS_FILE):
        try:
            with open(PROMPTS_FILE, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning(f"Error parsing {PROMPTS_FILE}. Starting with empty prompt list.")
            return []
    return []

def save_test_prompts(prompts):
    """Save test prompts to JSON file"""
    try:
        with open(PROMPTS_FILE, 'w') as f:
            json.dump(prompts, f, indent=4)
    except Exception as e:
        st.error(f"Error saving prompts: {str(e)}")

def get_parquet_path(vanilla_model_name):
    """Get path to parquet file for a vanilla model"""
    # Replace any characters that might be invalid in filenames
    safe_name = Path(vanilla_model_name).name
    return os.path.join(COMP_DATA_DIR, f"{safe_name}.parquet")

def load_comparison_data(vanilla_model_name):
    """Load comparison data from parquet file"""
    parquet_path = get_parquet_path(vanilla_model_name)
    
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            st.warning(f"Error loading parquet file: {str(e)}. Creating new dataframe.")
            return pd.DataFrame(columns=["prompt", "vanilla"])
    
    # Create empty dataframe with prompt column
    return pd.DataFrame(columns=["prompt", "vanilla"])

def save_comparison_data(df, vanilla_model_name):
    """Save comparison data to parquet file"""
    parquet_path = get_parquet_path(vanilla_model_name)
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")

def update_comparison_data(df, prompts, model_col, responses, times):
    """Update dataframe with new responses while preserving existing data"""
    # Make sure model_col exists in dataframe
    if model_col not in df.columns:
        df[model_col] = None
        df[f"{model_col}_time"] = None
        df[f"{model_col}_timestamp"] = None
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # For each prompt and response
    new_rows = []
    for prompt, response, gen_time in zip(prompts, responses, times):
        # Check if prompt exists in dataframe
        if prompt in df["prompt"].values:
            # Update existing row
            idx = df[df["prompt"] == prompt].index[0]
            df.at[idx, model_col] = response
            df.at[idx, f"{model_col}_time"] = gen_time
            df.at[idx, f"{model_col}_timestamp"] = timestamp
        else:
            # Create a new row
            new_row = {
                "prompt": prompt,
                model_col: response,
                f"{model_col}_time": gen_time,
                f"{model_col}_timestamp": timestamp
            }
            new_rows.append(new_row)
    
    # Append any new rows
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
    
    return df

# App title
st.title("LLM Comparison Tool")
st.write(f"Running on device: {device}")

# Scan for available models
with st.spinner("Scanning for available models..."):
    available_models = scan_models()

if not available_models:
    st.warning("No models found in the outputs directory. Please check your folder structure.")
    st.stop()

# Model selection
vanilla_model_options = list(available_models.keys())
selected_vanilla_model = st.selectbox("Select Vanilla Model", vanilla_model_options)

# Get finetunes for the selected vanilla model
finetune_options = available_models[selected_vanilla_model]
if not finetune_options:
    st.warning(f"No fine-tuned models found for {selected_vanilla_model}.")
    st.stop()

selected_finetune = st.selectbox("Select Fine-tuned Model", finetune_options)

# Load test prompts
test_prompts = load_test_prompts()

# Load comparison data
comparison_data = load_comparison_data(selected_vanilla_model)

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Prompt Management", "Grid View", "Side-by-Side View"])

with tab1:
    st.subheader("Manage Test Prompts")
    
    # Add new prompt
    with st.form("add_prompt_form"):
        new_prompt = st.text_area("Enter a new prompt:", height=100)
        submitted = st.form_submit_button("Add Prompt")
    
    if submitted and new_prompt:
        if new_prompt not in test_prompts:
            test_prompts.append(new_prompt)
            save_test_prompts(test_prompts)
            st.success("Prompt added successfully!")
        else:
            st.warning("This prompt already exists in the list.")
    
    # Display current prompts
    st.subheader("Current Test Prompts")
    
    if not test_prompts:
        st.info("No test prompts available. Add some prompts above.")
    else:
        for i, prompt in enumerate(test_prompts):
            col1, col2 = st.columns([0.9, 0.1])
            with col1:
                st.text_area(f"Prompt {i+1}", prompt, height=100, key=f"prompt_display_{i}", disabled=True, 
                            label_visibility="collapsed")
            with col2:
                if st.button("Delete", key=f"delete_prompt_{i}"):
                    test_prompts.pop(i)
                    save_test_prompts(test_prompts)
                    st.rerun()
    
    # Generate responses
    st.subheader("Generate Responses")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Find prompts without vanilla responses
        vanilla_missing = []
        for p in test_prompts:
            if p not in comparison_data["prompt"].values or pd.isna(comparison_data.loc[comparison_data["prompt"] == p, "vanilla"].values[0]):
                vanilla_missing.append(p)
        
        if vanilla_missing:
            st.write(f"Found {len(vanilla_missing)} prompts without vanilla model responses.")
        
        if st.button("Generate with Vanilla Model", disabled=len(vanilla_missing) == 0):
            with st.status("Processing with vanilla model...", expanded=True) as status:
                status.update(label=f"Loading vanilla model: {selected_vanilla_model}")
                model, tokenizer = load_vanilla_model(selected_vanilla_model)
                
                responses = []
                gen_times = []
                progress_bar = st.progress(0)
                
                for i, prompt in enumerate(vanilla_missing):
                    status.update(label=f"Processing prompt {i+1}/{len(vanilla_missing)}")
                    response, gen_time = generate_response(model, tokenizer, prompt)
                    responses.append(response)
                    gen_times.append(gen_time)
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(vanilla_missing))
                
                # Update comparison data
                comparison_data = update_comparison_data(
                    comparison_data, vanilla_missing, "vanilla", responses, gen_times
                )
                save_comparison_data(comparison_data, selected_vanilla_model)
                
                status.update(label=f"Processed {len(vanilla_missing)} prompts with vanilla model!", state="complete")
    
    with col2:
        # Find prompts without finetune responses
        finetune_col = selected_finetune
        
        # Find prompts that need finetune responses
        finetune_missing = []
        for p in test_prompts:
            # Check if prompt exists in dataframe
            if p not in comparison_data["prompt"].values:
                finetune_missing.append(p)
            # Check if finetune column exists and value is NA
            elif finetune_col not in comparison_data.columns or pd.isna(comparison_data.loc[comparison_data["prompt"] == p, finetune_col].values[0]):
                finetune_missing.append(p)
        
        if finetune_missing:
            st.write(f"Found {len(finetune_missing)} prompts without {selected_finetune} responses.")
        
        if st.button("Generate with Fine-tuned Model", disabled=len(finetune_missing) == 0):
            with st.status("Processing with fine-tuned model...", expanded=True) as status:
                status.update(label=f"Loading fine-tuned model: {selected_finetune}")
                model, tokenizer = load_finetuned_model(selected_vanilla_model, selected_finetune)
                
                responses = []
                gen_times = []
                progress_bar = st.progress(0)
                
                for i, prompt in enumerate(finetune_missing):
                    status.update(label=f"Processing prompt {i+1}/{len(finetune_missing)}")
                    response, gen_time = generate_response(model, tokenizer, prompt)
                    responses.append(response)
                    gen_times.append(gen_time)
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(finetune_missing))
                
                # Update comparison data
                comparison_data = update_comparison_data(
                    comparison_data, finetune_missing, finetune_col, responses, gen_times
                )
                save_comparison_data(comparison_data, selected_vanilla_model)
                
                status.update(label=f"Processed {len(finetune_missing)} prompts with {selected_finetune}!", state="complete")

with tab2:
    st.subheader("Grid View")
    
    if comparison_data.empty:
        st.info("No comparison data available yet. Generate some responses first.")
    else:
        # Filter to only show prompts in the test_prompts list
        filtered_data = comparison_data[comparison_data["prompt"].isin(test_prompts)]
        
        if filtered_data.empty:
            st.info("No data available for the current test prompts.")
        else:
            # Create a display dataframe with just the columns we want to show
            display_cols = ["prompt", "vanilla"]
            
            # Add any finetune columns that exist
            for col in filtered_data.columns:
                if col not in ["prompt", "vanilla"] and not col.endswith("_time") and not col.endswith("_timestamp"):
                    display_cols.append(col)
            
            # Only include columns that actually exist
            display_cols = [col for col in display_cols if col in filtered_data.columns]
            
            st.dataframe(
                filtered_data[display_cols],
                use_container_width=True,
                column_config={
                    "prompt": st.column_config.TextColumn("Prompt", width="medium"),
                    "vanilla": st.column_config.TextColumn("Vanilla Model", width="large"),
                    **{col: st.column_config.TextColumn(col, width="large") for col in display_cols if col not in ["prompt", "vanilla"]}
                }
            )

with tab3:
    st.subheader("Side-by-Side View")
    
    if comparison_data.empty:
        st.info("No comparison data available yet. Generate some responses first.")
    else:
        # Filter to only show prompts in the test_prompts list
        filtered_data = comparison_data[comparison_data["prompt"].isin(test_prompts)]
        
        if filtered_data.empty:
            st.info("No data available for the current test prompts.")
        else:
            # Create a prompt selector
            prompt_options = filtered_data["prompt"].tolist()
            
            # Format function to truncate long prompts
            def format_prompt(prompt):
                return f"{prompt[:50]}..." if len(prompt) > 50 else prompt
            
            if prompt_options:
                selected_prompt_idx = st.selectbox(
                    "Select Prompt", 
                    range(len(prompt_options)), 
                    format_func=lambda i: format_prompt(prompt_options[i])
                )
                
                selected_prompt = prompt_options[selected_prompt_idx]
                prompt_row = filtered_data[filtered_data["prompt"] == selected_prompt].iloc[0]
                
                st.write("---")
                
                # Display the prompt
                st.markdown("### Prompt")
                st.text_area("", selected_prompt, height=150, disabled=True, label_visibility="collapsed")
                
                # Create columns for responses
                cols = st.columns(2)
                
                # Display vanilla response
                with cols[0]:
                    st.markdown("### Vanilla Model")
                    if "vanilla" in prompt_row and pd.notna(prompt_row["vanilla"]):
                        st.text_area("", prompt_row["vanilla"], height=400, disabled=True, label_visibility="collapsed")
                        if "vanilla_time" in prompt_row and pd.notna(prompt_row["vanilla_time"]):
                            st.caption(f"Generation time: {prompt_row['vanilla_time']:.2f} seconds")
                        if "vanilla_timestamp" in prompt_row and pd.notna(prompt_row["vanilla_timestamp"]):
                            st.caption(f"Generated: {prompt_row['vanilla_timestamp']}")
                    else:
                        st.info("No response generated yet.")
                
                # Display finetune response
                with cols[1]:
                    st.markdown(f"### Fine-tuned Model: {selected_finetune}")
                    if selected_finetune in prompt_row and pd.notna(prompt_row[selected_finetune]):
                        st.text_area("", prompt_row[selected_finetune], height=400, disabled=True, label_visibility="collapsed")
                        
                        # Check for training indicators if this is the GSM8K model
                        if "gsm8k" in selected_finetune.lower():
                            has_reasoning_tags = "<start_working_out>" in prompt_row[selected_finetune] and "<end_working_out>" in prompt_row[selected_finetune]
                            has_solution_tags = "<SOLUTION>" in prompt_row[selected_finetune] and "</SOLUTION>" in prompt_row[selected_finetune]
                            
                            if has_reasoning_tags and has_solution_tags:
                                st.success("✓ Following trained format")
                            elif has_reasoning_tags or has_solution_tags:
                                st.warning("~ Partially following trained format")
                            else:
                                st.error("✗ Not following trained format")
                        
                        time_col = f"{selected_finetune}_time"
                        timestamp_col = f"{selected_finetune}_timestamp"
                        
                        if time_col in prompt_row and pd.notna(prompt_row[time_col]):
                            st.caption(f"Generation time: {prompt_row[time_col]:.2f} seconds")
                        if timestamp_col in prompt_row and pd.notna(prompt_row[timestamp_col]):
                            st.caption(f"Generated: {prompt_row[timestamp_col]}")
                    else:
                        st.info("No response generated yet.")
            else:
                st.info("No prompts available for comparison.")