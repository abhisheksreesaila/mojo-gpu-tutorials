import os
import sys
from huggingface_hub import HfApi, SpaceHardware

# --- CONFIGURATION ---
# 1. Get Token (Required)
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    # Fallback: You can hardcode it here for testing if you prefer, but Env Var is safer
    raise ValueError("❌ HF_TOKEN environment variable not set. Run: export HF_TOKEN='hf_...'")

# 2. Get Repo ID (Default provided based on your previous chat)
# IMPORTANT: This must match the ID on HuggingFace exactly (usually underscores)
REPO_ID = os.getenv("HF_REPO_ID", "asreesaila/mojo_monday")

# Initialize API
api = HfApi(token=HF_TOKEN)

def start_mission():
    print(f"🚀 Launching Space: {REPO_ID} on A10G...")
    try:
        api.request_space_hardware(repo_id=REPO_ID, hardware=SpaceHardware.A10G_SMALL)
        print("✅ Signal sent. Space is building/starting.")
        print("⏳ Wait 2-3 minutes, then run 'status'.")
    except Exception as e:
        print(f"❌ Failed to start: {e}")

def stop_mission():
    print(f"🛑 Aborting Mission (Pausing Space)...")
    try:
        api.pause_space(repo_id=REPO_ID)
        print("✅ Space paused. Billing stopped.")
    except Exception as e:
        print(f"❌ Failed to stop: {e}")

def status_report():
    print(f"🔍 Checking status for: {REPO_ID}...")
    try:
        runtime = api.get_space_runtime(repo_id=REPO_ID)
        print(f"--------------------------------")
        print(f"📊 Stage:    {runtime.stage}")     # RUNNING, PAUSED, BUILDING
        print(f"💻 Hardware: {runtime.hardware}")  # cpu-basic, a10g-small
        print(f"--------------------------------")
        
        if runtime.stage == "RUNNING":
            print("💰 BILLING IS ACTIVE. Remember to stop when done!")
        elif runtime.stage == "PAUSED":
            print("zzz Space is sleeping (No Cost).")
            
    except Exception as e:
        print(f"❌ Error fetching status: {e}")
        print(f"   (Check if '{REPO_ID}' is the correct ID)")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\nUsage: python control-hf-spaces.py [start|stop|status]")
        print(f"Current Target Repo: {REPO_ID}")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "start":
        start_mission()
    elif command == "stop":
        stop_mission()
    elif command == "status":
        status_report()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python control-hf-spaces.py [start|stop|status]")