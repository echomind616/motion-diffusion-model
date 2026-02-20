import runpod
import os
import subprocess
import torch
from datetime import datetime

# NOTE: This handler is designed for MDM (Motion Diffusion Model)
# It generates a .npy motion file and then converts it to FBX using Blender.

def handler(job):
    """
    The handler function that will be called by Runpod.
    """
    job_input = job['input']
    prompt = job_input.get('prompt', 'a person is walking')
    
    print(f"🎬 Received job: {job['id']}")
    print(f"📝 Prompt: {prompt}")
    
    try:
        # Step 1: Run MDM Inference
        # Based on GuyTevet/motion-diffusion-model
        print("🤖 Running MDM inference...")
        
        # We assume the weights are in ./save/humanml_trans_enc_512
        # You can override this via env vars or job input
        model_path = os.environ.get("MDM_MODEL_PATH", "./save/humanml_trans_enc_512")
        
        # Command to run MDM
        inference_cmd = [
            "python", "-m", "sample.generate",
            "--model_path", model_path,
            "--text_prompt", prompt,
            "--num_samples", "1",
            "--num_repetitions", "1",
            "--output_dir", f"./jobs/{job['id']}"
        ]
        
        subprocess.run(inference_cmd, check=True)
        
        # MDM saves results as results.npy in the output_dir
        npy_path = f"./jobs/{job['id']}/results.npy"
        
        # Step 2: Convert to FBX using Headless Blender
        print("📦 Converting to FBX...")
        fbx_path = f"output_{job['id']}.fbx"
        
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"MDM inference failed to produce {npy_path}")

        conversion_cmd = [
            "blender",
            "--background",
            "--python", "motion_to_fbx.py",
            "--",
            "--input", npy_path,
            "--output", fbx_path
        ]
        
        try:
            subprocess.run(conversion_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ FBX Conversion failed: {e}")
            # Fallback or detailed error
            return {
                "status": "partial_success",
                "message": "Motion generated but FBX conversion failed. Check Blender installation and SMPL models.",
                "npy_path": npy_path,
                "prompt": prompt
            }

        # Step 3: Read and Return as Base64
        if os.path.exists(fbx_path):
            import base64
            with open(fbx_path, "rb") as f:
                fbx_data = base64.b64encode(f.read()).decode('utf-8')
            
            return {
                "status": "completed",
                "fbx_base64": fbx_data,
                "filename": fbx_path,
                "prompt": prompt
            }
        else:
            return {
                "status": "error",
                "message": "FBX file was not created by Blender."
            }

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
