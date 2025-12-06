"""
Gradio UI for Text-to-Video Generation
Run: python app.py
"""
import gradio as gr
import torch
from inference import VideoGenerator, find_latest_checkpoint
import os
from pathlib import Path

# Global generator (loaded once)
generator = None

def load_generator():
    """Load the video generator (lazy loading)"""
    global generator
    if generator is None:
        checkpoint = find_latest_checkpoint()
        if checkpoint:
            print(f"Loading checkpoint: {checkpoint}")
        generator = VideoGenerator(checkpoint_path=checkpoint)
    return generator

def generate_video(prompt, num_steps, guidance_scale):
    """Generate video from prompt"""
    try:
        gen = load_generator()
        
        if not prompt or len(prompt.strip()) == 0:
            return None, "❌ Please enter a text prompt!"
        
        print(f"Generating video: '{prompt}'")
        video = gen.generate(
            prompt=prompt,
            num_inference_steps=int(num_steps),
            guidance_scale=guidance_scale
        )
        
        # Save temporary video
        output_path = f"outputs/generated_{hash(prompt) % 10000}.mp4"
        os.makedirs("outputs", exist_ok=True)
        gen.save_video(video, output_path, fps=8)
        
        return output_path, f"✅ Video generated! Saved to {output_path}"
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg

# Create Gradio interface
def create_interface():
    """Create Gradio interface"""
    
    # Check if checkpoint exists
    checkpoint = find_latest_checkpoint()
    if checkpoint:
        status_msg = f"✅ Model loaded from: {checkpoint}"
    else:
        status_msg = "⚠️  No checkpoint found. Model will use random weights."
    
    # Older Gradio versions do not support the `theme` argument on Blocks
    with gr.Blocks(title="Text-to-Video Generator") as demo:
        gr.Markdown(
            """
            # 🎬 Text-to-Video Generator
            
            Generate videos from text descriptions using a Transformer-based diffusion model.
            
            **Instructions:**
            1. Enter a text prompt describing the video you want to generate
            2. Adjust inference steps (more steps = better quality, slower)
            3. Click "Generate Video"
            4. Wait for the video to be generated (30-60 seconds)
            
            **Example prompts:**
            - "A cat playing with a ball"
            - "A person walking in a park"
            - "Ocean waves crashing on the beach"
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Text Prompt",
                    placeholder="Enter a description of the video you want to generate...",
                    lines=3,
                    value="A cat playing with a ball"
                )
                
                with gr.Row():
                    num_steps = gr.Slider(
                        label="Inference Steps",
                        minimum=20,
                        maximum=100,
                        value=50,
                        step=10,
                        info="More steps = better quality, slower generation"
                    )
                    
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=15.0,
                        value=7.5,
                        step=0.5,
                        info="How closely to follow the prompt"
                    )
                
                generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
                status_output = gr.Textbox(label="Status", interactive=False)
            
            with gr.Column(scale=1):
                video_output = gr.Video(label="Generated Video")
        
        gr.Markdown(f"**Status:** {status_msg}")
        
        # Event handlers
        generate_btn.click(
            fn=generate_video,
            inputs=[prompt_input, num_steps, guidance_scale],
            outputs=[video_output, status_output]
        )
        
        # Example prompts
        gr.Examples(
            examples=[
                ["A cat playing with a ball"],
                ["A person walking in a park"],
                ["Ocean waves crashing on the beach"],
                ["A bird flying in the sky"],
                ["A car driving on a highway"],
            ],
            inputs=prompt_input
        )
    
    return demo

if __name__ == "__main__":
    # Create outputs directory
    os.makedirs("outputs", exist_ok=True)
    
    # Create and launch interface
    demo = create_interface()
    # Allow overriding the port / share settings via environment variables
    server_port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share_flag = os.getenv("GRADIO_SHARE", "true").lower() == "true"
    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=share_flag
    )

