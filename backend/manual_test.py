import requests
import json

API_URL = 'http://127.0.0.1:5000/detect'

def test_attack(name, sensor_data, expected):
    """Test a single attack scenario"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    response = requests.post(API_URL, json={'sensor_data': sensor_data})
    result = response.json()
    
    print(f"Reconstruction Error: {result['reconstruction_error']:.6f}")
    print(f"Threshold:           {result['threshold']:.6f}")
    print(f"Attack Detected:     {result['attack_detected']}")
    print(f"Status:              {result['status']}")
    
    # Check if result matches expectation
    if result['attack_detected'] == expected:
        print(f"✅ PASS - {'Detected' if expected else 'Not detected'} as expected")
    else:
        print(f"❌ FAIL - Expected {expected}, got {result['attack_detected']}")
    
    return result

# Test Suite
print("\n🧪 MANUAL DETECTION TEST SUITE")
print("="*60)

# Test 1: EXTREME Attack (Tank Overflow)
extreme = [0.0, 1.0, 0.0, 0.35, 0.98, 0.71, 1.0, 0.01, 0.0, 0.50, 0.0, 0.0, 0.0, 0.0]
test_attack("EXTREME - Tank Overflow Attack", extreme, expected=True)

# Test 2: MEDIUM Attack (Flow Manipulation)
medium = [0.15, 0.92, 0.08, 0.45, 0.75, 0.85, 0.18, 0.62, 0.25, 0.15, 0.80, 0.72, 0.28, 0.55]
test_attack("MEDIUM - Flow Manipulation", medium, expected=True)

# Test 3: SUBTLE Attack (Sensor Drift)
subtle = [0.48, 0.82, 0.15, 0.92, 0.68, 0.71, 0.25, 0.58, 0.88, 0.22, 0.74, 0.69, 0.35, 0.51]
test_attack("SUBTLE - Sensor Drift", subtle, expected=False)  # May or may not detect

# Test 4: VERY SUBTLE (Minimal Deviation)
very_subtle = [0.50, 0.55, 0.12, 0.88, 0.67, 0.38, 0.22, 0.54, 0.91, 0.19, 0.75, 0.65, 0.30, 0.48]
test_attack("VERY SUBTLE - Minimal Deviation", very_subtle, expected=False)

# Test 5: NORMAL Operation
normal = [0.52, 0.44, 0.11, 0.87, 0.65, 0.33, 0.21, 0.55, 0.92, 0.18, 0.76, 0.66, 0.31, 0.49]
test_attack("NORMAL - Baseline Operation", normal, expected=False)

# Test 6: Critical Tank Level
critical_tank = [0.12, 0.95, 0.10, 0.85, 0.70, 0.88, 0.20, 0.60, 0.30, 0.20, 0.75, 0.70, 0.25, 0.50]
test_attack("CRITICAL - Tank Near Overflow", critical_tank, expected=True)

# Summary
print("\n" + "="*60)
print("📊 TEST SUMMARY")
print("="*60)
print("\n✅ All tests completed!")
print("\nExpected Results:")
print("  EXTREME:      Should DETECT (error > 0.15)")
print("  MEDIUM:       Should DETECT (error > 0.10)")
print("  SUBTLE:       May/may not detect (error ~0.05)")
print("  VERY SUBTLE:  Should NOT detect (error < 0.05)")
print("  NORMAL:       Should NOT detect (error < 0.01)")
print("="*60)