// src/services/api.js - UPDATED VERSION
// Copy-paste this to replace your current api.js

const API_BASE = 'http://127.0.0.1:5000';

export const detectAnomaly = async (sensorValues) => {
  try {
    console.log('🔍 Calling /detect with', sensorValues.length, 'sensors');
    
    const response = await fetch(`${API_BASE}/detect`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sensor_data: sensorValues })
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('❌ /detect error:', errorText);
      throw new Error(`API Error: ${response.status}`);
    }
    
    const result = await response.json();
    console.log('✅ /detect result:', result);
    return result;
    
  } catch (error) {
    console.error('API Call Failed:', error);
    throw error;
  }
};

export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return await response.json();
  } catch (error) {
    console.error('Health Check Failed:', error);
    return { status: 'offline' };
  }
};

export const generateAttackPattern = async (attackType, targetStage) => {
  try {
    console.log(`🤖 Calling GenAI to generate ${attackType}...`);
    
    const response = await fetch(`${API_BASE}/genai/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        attack_type: attackType,
        target_stage: targetStage
      })
    });
    
    if (!response.ok) {
      throw new Error(`GenAI API Error: ${response.status}`);
    }
    
    const data = await response.json();
    console.log('✓ GenAI generated attack:', data.attack_pattern.slice(0, 5), '...');
    
    return {
      attack_vector: data.attack_pattern,
      type: data.type,
      target: data.target
    };
    
  } catch (error) {
    console.error('❌ GenAI generation failed:', error);
    
    // Fallback pattern
    console.warn('⚠️ Using fallback pattern');
    return {
      attack_vector: [0.52, 0.44, 0.11, 0.87, 0.65, 0.33, 0.21, 0.55, 0.92, 0.18, 0.76, 0.66, 0.31, 0.49],
      type: attackType,
      target: targetStage
    };
  }
};

export const simulateAttack = async (sensorValues, steps = 60) => {
  try {
    const response = await fetch(`${API_BASE}/twin/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        sensor_data: sensorValues,
        steps: steps
      })
    });
    
    if (!response.ok) {
      throw new Error(`Simulation API Error: ${response.status}`);
    }
    
    return await response.json();
    
  } catch (error) {
    console.error('Simulation failed:', error);
    throw error;
  }
};

export const discoverGaps = async (attackPattern) => {
  try {
    console.log('🔍 Discovering cybersecurity gaps...');
    
    const response = await fetch(`${API_BASE}/gaps/discover`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        attack_pattern: attackPattern
      })
    });
    
    if (!response.ok) {
      throw new Error(`Gap Discovery API Error: ${response.status}`);
    }
    
    const data = await response.json();
    console.log(`✓ Discovered ${data.count} vulnerabilities`);
    
    return data.gaps;
    
  } catch (error) {
    console.error('Gap discovery failed:', error);
    throw error;
  }
};

// Get normal sensor values
export const getNormalPattern = () => {
  return [0.52, 0.44, 0.11, 0.87, 0.65, 0.33, 0.21, 0.55, 0.92, 0.18, 0.76, 0.66, 0.31, 0.49];
};