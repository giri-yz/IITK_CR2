#!/usr/bin/env python3
"""
Quick test script for GenAI integration
Run this to verify everything works before integrating into main app
"""
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from genai_vulnerability_scanner import LLMVulnerabilityScanner
import json


def test_groq_connection():
    """Test if Groq API key is set and working"""   
    print("=" * 70)
    print("TEST 1: Groq API Connection")
    print("=" * 70)
    
    api_key = os.getenv("GROQ_API_KEY")

    
    if not api_key:
        print("❌ GROQ_API_KEY not set")
        print("\nTo fix:")
        print("1. Get free API key from: https://console.groq.com")
        print("2. Set it: export GROQ_API_KEY='your_key_here'")
        print("3. Run this script again")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...{api_key[-5:]}")
    
    scanner = LLMVulnerabilityScanner()
    
    # Simple test
    try:
        result = scanner._call_llm("Say 'WORKING' if you can read this", temperature=0)
        if result and "WORKING" in result.upper():
            print("✅ API connection successful!")
            print(f"   Model: {scanner.model}")
            return True
        else:
            print(f"⚠️  Got response but unexpected: {result}")
            return False
    except Exception as e:
        print(f"❌ API Error: {e}")
        return False


def test_vulnerability_discovery():
    """Test vulnerability discovery"""
    print("\n" + "=" * 70)
    print("TEST 2: Vulnerability Discovery")
    print("=" * 70)
    
    scanner = LLMVulnerabilityScanner()
    
    sensor_data = {
        "FIT101": "Raw water flow sensor (0-2 m³/hr)",
        "LIT101": "Raw water tank level (0-1000 mm)",
        "FIT201": "Chemical dosing flow (0-2 m³/hr)",
        "AIT201": "Water conductivity (0-2000 µS/cm)",
        "P101": "Raw water pump",
        "MV101": "Motorized valve"
    }
    
    process_desc = """
Water treatment system with critical constraints:
- Tank levels must stay 200-900mm
- Flow balance must be maintained
- Conductivity must be 100-500 µS/cm
- Pumps cannot run with level < 250mm
"""
    
    print("\n🔍 Asking LLM to find vulnerabilities...")
    print("(This takes 5-10 seconds)\n")
    
    result = scanner.analyze_system_architecture(sensor_data, process_desc)
    
    vulnerabilities = result.get('vulnerabilities', [])
    
    if vulnerabilities:
        print(f"✅ Found {len(vulnerabilities)} vulnerabilities:\n")
        for i, vuln in enumerate(vulnerabilities, 1):
            print(f"{i}. {vuln.get('name', 'Unknown')}")
            print(f"   Severity: {vuln.get('severity', 'N/A')}")
            print(f"   Target: {vuln.get('target', 'N/A')}")
            print(f"   Impact: {vuln.get('impact', 'N/A')[:80]}...")
            print()
        return True
    else:
        print("❌ No vulnerabilities found")
        print(f"Raw response: {result.get('raw_analysis', 'None')[:200]}")
        return False


def test_attack_generation():
    """Test attack vector generation"""
    print("\n" + "=" * 70)
    print("TEST 3: Attack Vector Generation")
    print("=" * 70)
    
    scanner = LLMVulnerabilityScanner()
    
    # Example vulnerability
    vulnerability = {
        "name": "Tank Overflow Attack",
        "target": "LIT101",
        "mechanism": "Manipulate flow sensor to cause tank overflow",
        "severity": "HIGH"
    }
    
    sensor_ranges = {
        "FIT101": "0.0-1.0 (flow rate)",
        "LIT101": "0.0-1.0 (tank level)",
        "P101": "0.0 (OFF) or 1.0 (ON)"
    }
    
    print("\n🎯 Generating attack vector...")
    print("(This takes 5-10 seconds)\n")
    
    attack = scanner.generate_attack_vector(vulnerability, sensor_ranges)
    
    if attack and 'sensor_values' in attack:
        print("✅ Attack generated successfully:")
        print(f"   Sensor values: {attack['sensor_values'][:5]}... (showing first 5)")
        print(f"   Detection difficulty: {attack.get('detection_difficulty', 'N/A')}/10")
        print(f"   Evasion strategy: {attack.get('evasion_reasoning', 'N/A')[:100]}...")
        return True
    else:
        print("❌ Attack generation failed")
        print(f"Response: {attack}")
        return False


def test_explanation():
    """Test detection explanation"""
    print("\n" + "=" * 70)
    print("TEST 4: Detection Result Explanation")
    print("=" * 70)
    
    scanner = LLMVulnerabilityScanner()
    
    import numpy as np
    attack_vector = np.array([0.5, 0.85, 0.6, 0.4, 0.7, 0.8, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    
    print("\n💡 Asking LLM to explain why attack evaded detection...")
    print("(This takes 3-5 seconds)\n")
    
    explanation = scanner.explain_detection_failure(attack_vector, detected=False, severity="SUBTLE")
    
    if explanation:
        print("✅ Explanation generated:")
        print(f"\n{explanation}\n")
        return True
    else:
        print("❌ Explanation generation failed")
        return False


def main():
    """Run all tests"""
    print("\n" + "🚀 " * 20)
    print("GenAI Vulnerability Scanner - Test Suite")
    print("🚀 " * 20 + "\n")
    
    results = {
        "API Connection": False,
        "Vulnerability Discovery": False,
        "Attack Generation": False,
        "Explanation": False
    }
    
    # Test 1: Connection
    results["API Connection"] = test_groq_connection()
    
    if not results["API Connection"]:
        print("\n❌ Cannot proceed without API connection")
        print("Set GROQ_API_KEY and try again")
        sys.exit(1)
    
    # Test 2: Vulnerability Discovery
    results["Vulnerability Discovery"] = test_vulnerability_discovery()
    
    # Test 3: Attack Generation
    results["Attack Generation"] = test_attack_generation()
    
    # Test 4: Explanation
    results["Explanation"] = test_explanation()
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your GenAI integration is ready!")
        print("\nNext steps:")
        print("1. Copy genai_vulnerability_scanner.py to your backend/ folder")
        print("2. Copy genai_enhanced_backend.py to your backend/ folder")
        print("3. Add to app.py: from genai_enhanced_backend import genai_bp")
        print("4. Add to app.py: app.register_blueprint(genai_bp)")
        print("5. Restart your backend server")
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
    
    print("\n" + "🚀 " * 20 + "\n")


if __name__ == "__main__":
    main()
