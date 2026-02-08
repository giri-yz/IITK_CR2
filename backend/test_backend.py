"""
GenTwin Backend - Quick Test Script
Run this AFTER starting app_gentwin.py to verify everything works
"""
import requests
import json

API_BASE = 'http://127.0.0.1:5000'

print("="*60)
print("🧪 GenTwin Backend Test Suite")
print("="*60)
print()

# Test 1: Health Check
print("Test 1: Health Check")
print("-" * 60)
try:
    response = requests.get(f'{API_BASE}/health')
    data = response.json()
    print(f"✅ Status: {data['status']}")
    print(f"✅ Components: {json.dumps(data['components'], indent=2)}")
except Exception as e:
    print(f"❌ Health check failed: {e}")
print()

# Test 2: VAE Detection (Normal)
print("Test 2: VAE Detection - Normal Pattern")
print("-" * 60)
try:
    normal_pattern = [0.52, 0.44, 0.11, 0.87, 0.65, 0.33, 0.21, 0.55, 0.92, 0.18, 0.76, 0.66, 0.31, 0.49]
    response = requests.post(
        f'{API_BASE}/detect',
        json={'sensor_data': normal_pattern}
    )
    data = response.json()
    print(f"✅ Attack Detected: {data['attack_detected']}")
    print(f"✅ Reconstruction Error: {data['reconstruction_error']:.6f}")
    print(f"✅ Status: {data['status']}")
except Exception as e:
    print(f"❌ Detection failed: {e}")
print()

# Test 3: VAE Detection (Attack)
print("Test 3: VAE Detection - Attack Pattern")
print("-" * 60)
try:
    attack_pattern = [0.0, 1.0, 0.0, 0.35, 0.98, 0.71, 1.0, 0.01, 0.0, 0.50, 0.0, 0.0, 0.0, 0.0]
    response = requests.post(
        f'{API_BASE}/detect',
        json={'sensor_data': attack_pattern}
    )
    data = response.json()
    print(f"✅ Attack Detected: {data['attack_detected']}")
    print(f"✅ Reconstruction Error: {data['reconstruction_error']:.6f}")
    print(f"✅ Status: {data['status']}")
except Exception as e:
    print(f"❌ Detection failed: {e}")
print()

# Test 4: GenAI Attack Generation
print("Test 4: GenAI Attack Generation")
print("-" * 60)
try:
    response = requests.post(
        f'{API_BASE}/genai/generate',
        json={
            'attack_type': 'genai_attack_1',
            'target_stage': 'stage3'
        }
    )
    data = response.json()
    print(f"✅ Success: {data['success']}")
    print(f"✅ Attack Type: {data['type']}")
    print(f"✅ Target: {data['target']}")
    print(f"✅ Pattern (first 5): {data['attack_pattern'][:5]}")
except Exception as e:
    print(f"❌ GenAI generation failed: {e}")
print()

# Test 5: Physics Validation
print("Test 5: Physics Twin Validation")
print("-" * 60)
try:
    test_data = [0.15, 0.95, 0.22, 0.45, 0.67, 0.33, 0.21, 0.55, 0.10, 0.18, 0.76, 0.66, 0.31, 0.49]
    response = requests.post(
        f'{API_BASE}/physics/validate',
        json={'sensor_data': test_data}
    )
    data = response.json()
    print(f"✅ Is Valid: {data['is_valid']}")
    print(f"✅ Physics Score: {data['physics_score']:.2f}")
    print(f"✅ Violations: {data['violations']}")
    print(f"✅ Failure Mode: {data['failure_mode']}")
except Exception as e:
    print(f"❌ Physics validation failed: {e}")
print()

# Test 6: Gap Discovery
print("Test 6: Cybersecurity Gap Discovery")
print("-" * 60)
try:
    attack_pattern = [0.0, 1.0, 0.0, 0.35, 0.98, 0.71, 1.0, 0.01, 0.0, 0.50, 0.0, 0.0, 0.0, 0.0]
    response = requests.post(
        f'{API_BASE}/gaps/discover',
        json={'attack_pattern': attack_pattern}
    )
    data = response.json()
    print(f"✅ Success: {data['success']}")
    print(f"✅ Gaps Found: {data['count']}")
    for i, gap in enumerate(data['gaps'], 1):
        print(f"\n   Gap {i}:")
        print(f"   - Description: {gap['description']}")
        print(f"   - Severity: {gap['severity']}")
        print(f"   - Recommendation: {gap['recommendation']}")
except Exception as e:
    print(f"❌ Gap discovery failed: {e}")
print()

print("="*60)
print("✅ All Tests Complete!")
print("="*60)
print()
print("If all tests passed, your backend is working correctly!")
print("You can now start the frontend with: npm run dev")
print()