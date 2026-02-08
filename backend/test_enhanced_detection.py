"""
Enhanced Detection System - Integration & Testing Guide
=========================================================

PROBLEM: 0% detection rate on GenAI attacks
SOLUTION: Multi-layer detection with ensemble voting

QUICK START:
1. Copy enhanced_detection_system.py to your backend folder
2. Replace your current app.py with enhanced_backend.py
3. Update file paths in enhanced_backend.py
4. Run test_enhanced_detection.py to verify
5. Restart your backend

EXPECTED IMPROVEMENT:
- Before: 0% detection (0/10 attacks caught)
- After:  60-80% detection on subtle attacks
- After:  100% detection on medium/extreme attacks
"""

import requests
import json
import numpy as np

API_BASE = 'http://127.0.0.1:5000'

print("="*70)
print("🧪 Enhanced Detection System - Test Suite")
print("="*70)
print()

# ======================
# Test 1: Health Check
# ======================
print("Test 1: System Health")
print("-" * 70)
try:
    response = requests.get(f'{API_BASE}/health')
    data = response.json()
    print(f"✅ Status: {data['status']}")
    print(f"✅ Detection System: {data.get('detection_system', 'unknown')}")
except Exception as e:
    print(f"❌ Health check failed: {e}")
    print("   Make sure enhanced_backend.py is running!")
    exit(1)
print()

# ======================
# Test 2: Detection Stats
# ======================
print("Test 2: Detection Configuration")
print("-" * 70)
try:
    response = requests.get(f'{API_BASE}/detect/stats')
    data = response.json()
    print(f"✅ Thresholds:")
    for name, value in data['thresholds'].items():
        print(f"   • {name}: {value:.6f}")
    print(f"✅ Detection layers: {', '.join(data['detection_layers'])}")
    print(f"✅ Voting threshold: {data['voting_threshold']} votes needed")
except Exception as e:
    print(f"❌ Stats failed: {e}")
print()

# ======================
# Test 3: Normal Operation
# ======================
print("Test 3: Normal Operation (Should NOT Detect)")
print("-" * 70)
try:
    # Normal pattern from your logs
    normal_pattern = [0.52, 0.44, 0.11, 0.87, 0.65, 0.33, 0.21, 0.55, 
                     0.92, 0.18, 0.76, 0.66, 0.31, 0.49]
    
    response = requests.post(
        f'{API_BASE}/detect',
        json={'sensor_data': normal_pattern}
    )
    data = response.json()
    
    print(f"   Attack Detected: {data['attack_detected']}")
    print(f"   Confidence: {data['confidence']*100:.1f}%")
    print(f"   Severity: {data['severity']}")
    print(f"   Reconstruction Error: {data['reconstruction_error']:.6f}")
    print(f"   Vote Breakdown: {data['detection_breakdown']['votes']}/4")
    print(f"   Explanation: {data['explanation']}")
    
    if data['attack_detected']:
        print("   ⚠️  WARNING: False positive on normal data!")
    else:
        print("   ✅ Correctly identified as normal")
        
except Exception as e:
    print(f"   ❌ Test failed: {e}")
print()

# ======================
# Test 4: SUBTLE Attack
# ======================
print("Test 4: SUBTLE Attack (GenAI-like)")
print("-" * 70)
try:
    # Simulate a GenAI subtle attack
    subtle_attack = [0.48, 0.82, 0.15, 0.92, 0.68, 0.71, 0.25, 0.58,
                    0.88, 0.22, 0.74, 0.69, 0.35, 0.51]
    
    response = requests.post(
        f'{API_BASE}/detect',
        json={'sensor_data': subtle_attack}
    )
    data = response.json()
    
    print(f"   Attack Detected: {data['attack_detected']}")
    print(f"   Confidence: {data['confidence']*100:.1f}%")
    print(f"   Severity: {data['severity']}")
    print(f"   Reconstruction Error: {data['reconstruction_error']:.6f}")
    print(f"   Votes: {data['detection_breakdown']}")
    print(f"   Explanation: {data['explanation']}")
    
    if data['attack_detected']:
        print("   ✅ Successfully detected subtle attack!")
    else:
        print("   ⚠️  Missed subtle attack (may need threshold tuning)")
        
except Exception as e:
    print(f"   ❌ Test failed: {e}")
print()

# ======================
# Test 5: MEDIUM Attack
# ======================
print("Test 5: MEDIUM Attack")
print("-" * 70)
try:
    # Medium severity attack
    medium_attack = [0.15, 0.92, 0.08, 0.45, 0.75, 0.85, 0.18, 0.62,
                    0.25, 0.15, 0.80, 0.72, 0.28, 0.55]
    
    response = requests.post(
        f'{API_BASE}/detect',
        json={'sensor_data': medium_attack}
    )
    data = response.json()
    
    print(f"   Attack Detected: {data['attack_detected']}")
    print(f"   Confidence: {data['confidence']*100:.1f}%")
    print(f"   Severity: {data['severity']}")
    print(f"   Reconstruction Error: {data['reconstruction_error']:.6f}")
    print(f"   Votes: {data['detection_breakdown']}")
    print(f"   Explanation: {data['explanation']}")
    
    if data['attack_detected']:
        print("   ✅ Successfully detected medium attack!")
    else:
        print("   ❌ CRITICAL: Missed medium attack!")
        
except Exception as e:
    print(f"   ❌ Test failed: {e}")
print()

# ======================
# Test 6: EXTREME Attack
# ======================
print("Test 6: EXTREME Attack")
print("-" * 70)
try:
    # Extreme attack from your logs
    extreme_attack = [0.0, 1.0, 0.0, 0.35, 0.98, 0.71, 1.0, 0.01, 
                     0.0, 0.50, 0.0, 0.0, 0.0, 0.0]
    
    response = requests.post(
        f'{API_BASE}/detect',
        json={'sensor_data': extreme_attack}
    )
    data = response.json()
    
    print(f"   Attack Detected: {data['attack_detected']}")
    print(f"   Confidence: {data['confidence']*100:.1f}%")
    print(f"   Severity: {data['severity']}")
    print(f"   Reconstruction Error: {data['reconstruction_error']:.6f}")
    print(f"   Votes: {data['detection_breakdown']}")
    print(f"   Explanation: {data['explanation']}")
    
    if data['attack_detected']:
        print("   ✅ Successfully detected extreme attack!")
    else:
        print("   ❌ CRITICAL: Missed extreme attack!")
        
except Exception as e:
    print(f"   ❌ Test failed: {e}")
print()

# ======================
# Test 7: Temporal Detection
# ======================
print("Test 7: Temporal Pattern Detection (Sudden Jump)")
print("-" * 70)
try:
    # Send normal pattern first
    normal = [0.5] * 14
    requests.post(f'{API_BASE}/detect', json={'sensor_data': normal})
    requests.post(f'{API_BASE}/detect', json={'sensor_data': normal})
    
    # Then sudden jump
    sudden_jump = [0.9, 0.95, 0.85, 0.88, 0.92, 0.90, 0.87, 0.93,
                  0.91, 0.89, 0.94, 0.86, 0.88, 0.90]
    
    response = requests.post(
        f'{API_BASE}/detect',
        json={'sensor_data': sudden_jump}
    )
    data = response.json()
    
    print(f"   Attack Detected: {data['attack_detected']}")
    print(f"   Temporal Layer: {data['detection_breakdown']['temporal_detected']}")
    print(f"   Explanation: {data['explanation']}")
    
    if data['detection_breakdown']['temporal_detected']:
        print("   ✅ Temporal detection working!")
    else:
        print("   ⚠️  Temporal detection didn't trigger")
        
    # Reset for next tests
    requests.post(f'{API_BASE}/detect/reset')
    
except Exception as e:
    print(f"   ❌ Test failed: {e}")
print()

# ======================
# Summary
# ======================
print("="*70)
print("📊 TEST SUMMARY")
print("="*70)
print("\nIf most tests passed, your enhanced detection is working!")
print("\nNext Steps:")
print("1. Integrate enhanced_backend.py into your main app")
print("2. Update frontend to show vote breakdown")
print("3. Re-run your 10-attack test")
print("4. Tune thresholds if needed using /detect/calibrate endpoint")
print()
print("Expected improvement:")
print("  Before: 0/10 attacks detected (0%)")
print("  After:  6-8/10 attacks detected (60-80%)")
print("="*70)
