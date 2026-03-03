"""
Run this script first to confirm the webhook is working BEFORE running face_system.py
Usage: python test_webhook.py
"""
import requests

MAKE_WEBHOOK_URL = "https://hook.eu1.make.com/qj676s4s6r0gipdpyfa39xxtbvus73b9"

print("Testing Make webhook...")
print(f"URL: {MAKE_WEBHOOK_URL}")

try:
    resp = requests.post(
        MAKE_WEBHOOK_URL,
        data="Aryan,2026-03-03 10:00:00\nAryan,2026-03-03 10:00:10",
        headers={"Content-Type": "text/plain"},
        timeout=15
    )
    print(f"SUCCESS - HTTP {resp.status_code}")
    print(f"Response: {resp.text}")
except requests.exceptions.ConnectionError as e:
    print(f"FAILED - Connection error: {e}")
    print("Check your internet connection or firewall settings.")
except requests.exceptions.Timeout:
    print("FAILED - Request timed out (Make server not responding)")
except Exception as e:
    print(f"FAILED - {type(e).__name__}: {e}")
