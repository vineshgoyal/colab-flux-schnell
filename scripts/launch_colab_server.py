#!/usr/bin/env python3
"""
Launch script for running the AI Image Studio FastAPI backend and Cloudflare tunnel on Google Colab.
"""
import os
import sys
import time
import re
import subprocess
import urllib.request
import json

def main():
    print("🚀 Initializing AI Image Studio Backend Server...")
    
    STUDIO_API_TOKEN = os.getenv("STUDIO_API_TOKEN") or "vinesh123"
    os.environ["STUDIO_API_TOKEN"] = STUDIO_API_TOKEN
    os.environ["STUDIO_AUTH_TOKEN"] = STUDIO_API_TOKEN
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Identify app dir
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print(f"📁 Application Directory: {app_dir}")
    
    # Kill any stale server/tunnel processes
    subprocess.run(["pkill", "-9", "-f", "uvicorn"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "cloudflared"], stderr=subprocess.DEVNULL)
    time.sleep(1.5)
    
    # Ensure bitsandbytes is installed for 4-bit FLUX quantization
    try:
        import bitsandbytes
        print("✅ bitsandbytes is installed.")
    except ImportError:
        print("📦 Installing bitsandbytes, sentencepiece, protobuf for 4-bit FLUX...")
        subprocess.run(["pip", "install", "-q", "bitsandbytes>=0.43.0", "sentencepiece", "protobuf"], check=True)
    
    # Download cloudflared binary if not present
    cloudflared_path = "/content/cloudflared-linux-amd64"
    if not os.path.exists(cloudflared_path):
        print("⬇️ Downloading cloudflared binary...")
        subprocess.run([
            "wget", "-q", "-nc",
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
            "-O", cloudflared_path
        ], check=True)
        subprocess.run(["chmod", "+x", cloudflared_path], check=True)
    
    # Start Uvicorn
    print("▶️ Starting Uvicorn backend on port 8000...")
    server_log_file = "/content/server.log"
    server_log = open(server_log_file, "w")
    env = dict(os.environ, PYTHONPATH=f"{app_dir}:" + os.environ.get("PYTHONPATH", ""))
    server_proc = subprocess.Popen(
        ["uvicorn", "backend.server:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=app_dir,
        env=env,
        stdout=server_log,
        stderr=subprocess.STDOUT
    )
    
    # Wait for local health check
    health_ok = False
    for attempt in range(20):
        time.sleep(0.5)
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/health")
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode())
                    print(f"✅ Local backend healthy: {data.get('status')}")
                    health_ok = True
                    break
        except Exception:
            pass
            
    if not health_ok:
        print("⚠️ Warning: Backend health check took longer than expected. Checking log:")
        if os.path.exists(server_log_file):
            with open(server_log_file) as f:
                print(f.read()[-500:])
                
    # Start Cloudflare Tunnel
    print("🌐 Starting Cloudflare Tunnel...")
    tunnel_log_file = "/content/tunnel.log"
    tunnel_log = open(tunnel_log_file, "w")
    tunnel_proc = subprocess.Popen(
        [cloudflared_path, "tunnel", "--url", "http://127.0.0.1:8000"],
        stdout=tunnel_log,
        stderr=subprocess.STDOUT
    )
    
    # Extract public tunnel URL
    tunnel_url = None
    for _ in range(45):
        time.sleep(1)
        if os.path.exists(tunnel_log_file):
            with open(tunnel_log_file) as f:
                content = f.read()
                m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                if m:
                    tunnel_url = m.group(0)
                    break
                    
    if tunnel_url:
        print("\n" + "=" * 65)
        print("🎉 AI IMAGE STUDIO BACKEND IS LIVE!")
        print(f"👉 Public Tunnel URL: {tunnel_url}")
        print(f"🔑 Auth Token:        {STUDIO_API_TOKEN}")
        print("=" * 65 + "\n")
        
        # Save tunnel URL to file for easy retrieval
        with open("/content/tunnel_url.txt", "w") as f:
            f.write(tunnel_url + "\n")
            
        print("To verify from local environment, set in your .env:")
        print(f"STUDIO_API_URL={tunnel_url}")
        print(f"STUDIO_API_TOKEN={STUDIO_API_TOKEN}")
    else:
        print("❌ Failed to obtain Cloudflare tunnel URL. Tunnel log snippet:")
        if os.path.exists(tunnel_log_file):
            with open(tunnel_log_file) as f:
                print(f.read()[-500:])

if __name__ == "__main__":
    main()
