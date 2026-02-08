"""
Enhanced GenAI Attack Generator
Replaces random noise with real AI-powered attack generation
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Blueprint, request, jsonify
import numpy as np
import torch
from genai_vulnerability_scanner import LLMVulnerabilityScanner

# Create blueprint
genai_bp = Blueprint('genai', __name__)

# Initialize scanner
scanner = LLMVulnerabilityScanner()

# Cache for discovered vulnerabilities
VULNERABILITY_CACHE = {
    "vulnerabilities": [],
    "last_scan": None
}


@genai_bp.route("/api/genai/discover", methods=["POST"])
def discover_vulnerabilities():
    """
    Use LLM to discover vulnerabilities in the system
    This is the REAL GenAI component
    """
    try:
        data = request.json
        
        # Sensor information
        sensor_info = {
            "FIT101": "Raw water inflow sensor (0-2 m³/hr)",
            "LIT101": "Raw water tank level (0-1000 mm)",
            "FIT201": "Chemical dosing flow (0-2 m³/hr)",
            "AIT201": "Water conductivity (0-2000 µS/cm)",
            "FIT301": "UF feed flow (0-2 m³/hr)",
            "LIT301": "UF feed tank level (0-1000 mm)",
            "FIT401": "RO feed flow (0-2 m³/hr)",
            "LIT401": "RO feed tank level (0-1000 mm)",
            "P101": "Raw water pump status",
            "P201": "Chemical dosing pump",
            "P301": "UF pump",
            "P401": "RO pump",
            "MV101": "Motorized valve 1",
            "MV201": "Motorized valve 2"
        }
        
        process_description = """
Secure Water Treatment (SWaT) - 6 Stage Process:

Stage 1 (P1): Raw water intake and storage
- FIT101 monitors inflow, LIT101 monitors tank level
- P101 controls raw water pump
- Critical: Tank overflow if LIT101 > 900mm

Stage 2 (P2): Chemical dosing
- FIT201 controls chemical dosing flow
- AIT201 monitors water conductivity
- Critical: Wrong dosing affects water quality

Stage 3 (P3): Ultrafiltration (UF)
- FIT301 monitors UF feed flow
- LIT301 monitors UF tank level
- Critical: Pressure management

Stage 4 (P4): Dechlorination
- Prepares water for RO

Stage 5 (P5): Reverse Osmosis (RO)
- FIT401 monitors RO feed
- LIT401 monitors RO tank level
- Critical: Membrane damage from pressure spikes

Stage 6 (P6): Backwash cleaning
- System maintenance

PHYSICS CONSTRAINTS:
1. Flow balance: Inflow must match outflow ±10%
2. Tank levels: Must stay 200-900mm (safe zone)
3. Conductivity: Must be 100-500 µS/cm for RO
4. Pump coordination: Cannot run dry (level < 250mm)
5. Pressure: Flow rate changes cause pressure transients
"""
        
        print("🔍 Calling LLM to discover vulnerabilities...")
        
        # Call LLM for vulnerability analysis
        result = scanner.analyze_system_architecture(sensor_info, process_description)
        
        # Cache results
        VULNERABILITY_CACHE["vulnerabilities"] = result.get("vulnerabilities", [])
        VULNERABILITY_CACHE["last_scan"] = "now"
        
        # Format for frontend
        vulnerabilities = []
        for vuln in result.get("vulnerabilities", []):
            vulnerabilities.append({
                "id": len(vulnerabilities) + 1,
                "name": vuln.get("name", "Unknown"),
                "target": vuln.get("target", "SYSTEM"),
                "severity": vuln.get("severity", "MEDIUM"),
                "description": vuln.get("mechanism", ""),
                "impact": vuln.get("impact", ""),
                "evasion_strategy": vuln.get("evasion", ""),
                "exploitability": "HIGH" if vuln.get("severity") in ["CRITICAL", "HIGH"] else "MEDIUM"
            })
        
        return jsonify({
            "success": True,
            "vulnerabilities": vulnerabilities,
            "total": len(vulnerabilities),
            "analysis_method": "LLM-powered reasoning",
            "model": scanner.model
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e),
            "vulnerabilities": []
        }), 500


@genai_bp.route("/api/genai/generate-attack", methods=["POST"])
def generate_ai_attack():
    """
    Generate attack vector using LLM reasoning
    """
    try:
        data = request.json
        attack_type = data.get("attack_type", "genai_attack_1")
        target_stage = data.get("target_stage", "stage3")
        
        # Use cached vulnerabilities or generate generic
        vulnerabilities = VULNERABILITY_CACHE.get("vulnerabilities", [])
        
        if not vulnerabilities:
            # Fallback to rule-based if no LLM available
            print("⚠️  No cached vulnerabilities, using fallback")
            return generate_fallback_attack(target_stage)
        
        # Select a vulnerability to exploit
        import random
        vuln = random.choice(vulnerabilities)
        
        print(f"🎯 Generating attack for: {vuln.get('name', 'Unknown')}")
        
        # Sensor ranges (normalized 0-1)
        sensor_ranges = {
            "FIT101": "0.0-1.0 (flow rate)",
            "LIT101": "0.0-1.0 (tank level)",
            "FIT201": "0.0-1.0 (chemical flow)",
            "AIT201": "0.0-1.0 (conductivity)",
            "FIT301": "0.0-1.0 (UF flow)",
            "LIT301": "0.0-1.0 (UF tank)",
            "FIT401": "0.0-1.0 (RO flow)",
            "LIT401": "0.0-1.0 (RO tank)",
            "P101-P401": "0.0 (OFF) or 1.0 (ON)",
            "MV101-MV201": "0.0 (CLOSED) or 1.0 (OPEN)"
        }
        
        # Generate attack vector with LLM
        attack_params = scanner.generate_attack_vector(vuln, sensor_ranges)
        
        if attack_params and "sensor_values" in attack_params:
            sensor_values = attack_params["sensor_values"]
            
            # Ensure 14 values
            if len(sensor_values) < 14:
                sensor_values.extend([0.5] * (14 - len(sensor_values)))
            elif len(sensor_values) > 14:
                sensor_values = sensor_values[:14]
            
            return jsonify({
                "success": True,
                "attack_vector": sensor_values,
                "vulnerability_exploited": vuln.get("name", "Unknown"),
                "detection_difficulty": attack_params.get("detection_difficulty", 5),
                "execution_steps": attack_params.get("execution_steps", []),
                "evasion_reasoning": attack_params.get("evasion_reasoning", ""),
                "generation_method": "LLM-powered"
            })
        else:
            print("⚠️  LLM generation failed, using fallback")
            return generate_fallback_attack(target_stage)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


def generate_fallback_attack(target_stage: str):
    """
    Fallback attack generation when LLM unavailable
    Still smarter than pure random
    """
    # Stage-specific attack patterns based on physics
    stage_attacks = {
        "stage1": {  # Raw water intake
            "FIT101": np.random.uniform(0.85, 0.95),  # High inflow
            "LIT101": np.random.uniform(0.85, 0.95),  # Near overflow
        },
        "stage2": {  # Chemical dosing
            "FIT201": np.random.uniform(0.75, 0.85),
            "AIT201": np.random.uniform(0.15, 0.25),  # Wrong conductivity
        },
        "stage3": {  # Ultrafiltration
            "FIT301": np.random.uniform(0.7, 0.85),
            "LIT301": np.random.uniform(0.75, 0.90),
        },
        "stage4": {  # RO stage
            "FIT401": np.random.uniform(0.7, 0.85),
            "LIT401": np.random.uniform(0.80, 0.95),
        }
    }
    
    # Base normal operation
    attack_vector = [0.5] * 14
    
    # Apply stage-specific perturbations
    stage_config = stage_attacks.get(target_stage, stage_attacks["stage3"])
    
    # Sensor mapping
    sensor_map = {
        "FIT101": 0, "LIT101": 1,
        "FIT201": 2, "AIT201": 3,
        "FIT301": 4, "LIT301": 5,
        "FIT401": 6, "LIT401": 7,
        "P101": 8, "P201": 9,
        "P301": 10, "P401": 11,
        "MV101": 12, "MV201": 13
    }
    
    for sensor, value in stage_config.items():
        if sensor in sensor_map:
            attack_vector[sensor_map[sensor]] = value
    
    return jsonify({
        "success": True,
        "attack_vector": attack_vector,
        "generation_method": "physics-based fallback",
        "target_stage": target_stage
    })


@genai_bp.route("/api/genai/explain-result", methods=["POST"])
def explain_detection():
    """
    Use LLM to explain detection results
    """
    try:
        data = request.json
        attack_vector = np.array(data.get("attack_vector", []))
        detected = data.get("detected", False)
        severity = data.get("severity", "MEDIUM")
        
        explanation = scanner.explain_detection_failure(attack_vector, detected, severity)
        
        return jsonify({
            "success": True,
            "explanation": explanation
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@genai_bp.route("/api/genai/analyze-gaps", methods=["POST"])
def analyze_gaps():
    """
    Analyze detection results to find systematic gaps
    """
    try:
        data = request.json
        results = data.get("detection_results", [])
        
        analysis = scanner.discover_gaps(results)
        
        return jsonify({
            "success": True,
            "analysis": analysis
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# Health check
@genai_bp.route("/api/genai/status", methods=["GET"])
def status():
    """Check if GenAI is available"""
    has_api_key = scanner.api_key is not None
    
    return jsonify({
        "genai_available": has_api_key,
        "model": scanner.model if has_api_key else None,
        "provider": "Groq (free tier)",
        "capabilities": {
            "vulnerability_discovery": has_api_key,
            "attack_generation": has_api_key,
            "result_explanation": has_api_key,
            "gap_analysis": has_api_key,
            "fallback_mode": True
        }
    })


if __name__ == "__main__":
    print("GenAI Blueprint - Import this into your Flask app")
    print("\nUsage:")
    print("  from genai_enhanced_backend import genai_bp")
    print("  app.register_blueprint(genai_bp)")
