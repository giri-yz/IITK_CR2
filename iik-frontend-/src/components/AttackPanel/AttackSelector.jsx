import React, { useState } from 'react';
import { detectAnomaly, generateAttackPattern } from '../../services/api';
import './AttackSelector.css';

const AttackSelector = ({ onAttackInject }) => {
  const [selectedAttack, setSelectedAttack] = useState('genai_attack_1');
  const [selectedStage, setSelectedStage] = useState('stage3');
  const [isInjecting, setIsInjecting] = useState(false);
  const [lastResult, setLastResult] = useState(null);

  const attacks = [
    {
      id: 'sensor_spoofing',
      name: 'Sensor Spoofing',
      icon: '📡',
      description: 'Manipulate sensor readings',
      severity: 'medium',
      target: 'Sensor'
    },
    {
      id: 'plc_manipulation',
      name: 'PLC Manipulation',
      icon: '🔧',
      description: 'Override control logic',
      severity: 'high',
      target: 'PLC'
    },
    {
      id: 'genai_attack_1',
      name: 'GenAI Attack #1',
      icon: '🤖',
      description: 'AI-generated attack pattern',
      severity: 'critical',
      target: 'Multiple'
    },
    {
      id: 'overflow_attack',
      name: 'Overflow Attack',
      icon: '💧',
      description: 'Force tank overflow',
      severity: 'critical',
      target: 'Tank'
    },
    {
      id: 'contamination',
      name: 'Water Contamination',
      icon: '☢️',
      description: 'Inject contaminants',
      severity: 'critical',
      target: 'Chemical'
    },
    {
      id: 'denial_of_service',
      name: 'DoS Attack',
      icon: '🚫',
      description: 'Block system communication',
      severity: 'high',
      target: 'Network'
    }
  ];

  const stages = [
    { id: 'stage1', name: 'Stage 1: Raw Water' },
    { id: 'stage2', name: 'Stage 2: Chemical Dosing' },
    { id: 'stage3', name: 'Stage 3: Filtration' },
    { id: 'stage4', name: 'Stage 4: UV Treatment' },
    { id: 'stage5', name: 'Stage 5: pH Adjustment' },
    { id: 'stage6', name: 'Stage 6: Clean Water' }
  ];

const handleInject = async () => {
  console.log('🎯 Injecting Attack:', selectedAttack);
  console.log('🎯 Target Stage:', selectedStage);
  
  try {
    // 1. Generate attack with GenAI
    const attackData = await generateAttackPattern(selectedAttack, selectedStage);
    const attackVector = attackData.attack_vector;
    
    // 2. Detect anomaly with VAE
    const vaeResult = await detectAnomaly(attackVector);
    setLastResult(vaeResult);
    console.log('🤖 VAE Result:', vaeResult);
    
    // 3. Validate with physics twin
// 3. Validate with physics twin
const physicsResult = await fetch('http://localhost:5000/physics/validate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ sensor_data: attackVector })  // ✅ Changed from sensor_values
}).then(r => r.json());
    
    console.log('⚗️ Physics Twin:', physicsResult);
    
    console.log('═══════════════════════════════════');
    console.log('DETECTION RESULTS:');
    console.log(`VAE Detected: ${vaeResult.attack_detected ? '🚨 YES' : '✓ NO'}`);
    console.log(`Physics Violated: ${!physicsResult.is_valid ? '🚨 YES' : '✓ NO'}`);
    console.log(`FINAL: ${vaeResult.attack_detected || !physicsResult.is_valid ? '🚨 ATTACK' : '✓ NORMAL'}`);
    console.log('═══════════════════════════════════');
    
    // Trigger parent callback
    onAttackInject({
      attackType: selectedAttack,
      targetStage: selectedStage,
      vaeResult,
      physicsResult,
      sensorValues: attackVector
    });
    
    // 👇 RETURN THE RESULT!
    return {
      vae: vaeResult,
      physics: physicsResult,
      sensorValues: attackVector
    };
    
  } catch (error) {
    console.error('❌ Attack injection failed:', error);
    return null; // Return null on error
  }
};

  const selectedAttackData = attacks.find(a => a.id === selectedAttack);

  return (
    <div className="attack-selector">
      <div className="panel-header">
        <h3>Attack Injection</h3>
        <span className="panel-subtitle">Simulate cybersecurity attacks</span>
      </div>

      {/* Attack Type Selection */}
      <div className="attack-types">
        <h4 className="section-label">Select Attack Type</h4>
        <div className="attack-cards">
          {attacks.map(attack => (
            <label 
              key={attack.id}
              className={`attack-card ${selectedAttack === attack.id ? 'attack-card--selected' : ''}`}
            >
              <input
                type="radio"
                name="attack"
                value={attack.id}
                checked={selectedAttack === attack.id}
                onChange={(e) => setSelectedAttack(e.target.value)}
                hidden
              />
              <div className="attack-card__content">
                <div className="attack-icon">{attack.icon}</div>
                <div className="attack-info">
                  <h5>{attack.name}</h5>
                  <p>{attack.description}</p>
                </div>
                <div className="attack-meta">
                  <span className={`severity severity--${attack.severity}`}>
                    {attack.severity}
                  </span>
                  <span className="target">{attack.target}</span>
                </div>
              </div>
            </label>
          ))}
        </div>
      </div>

      {/* Target Stage Selection */}
      <div className="stage-selection">
        <h4 className="section-label">Target Stage</h4>
        <select 
          className="stage-select"
          value={selectedStage}
          onChange={(e) => setSelectedStage(e.target.value)}
        >
          {stages.map(stage => (
            <option key={stage.id} value={stage.id}>
              {stage.name}
            </option>
          ))}
        </select>
      </div>

      {/* Inject Button */}
      <button 
        className="inject-button"
        onClick={handleInject}
        disabled={isInjecting}
      >
        {isInjecting ? (
          <>
            <div className="button-spinner"></div>
            <span className="button-text">Injecting...</span>
          </>
        ) : (
          <>
            <span className="button-icon">⚠️</span>
            <span className="button-text">Inject Attack</span>
          </>
        )}
      </button>

      {/* Add this AFTER the inject button */}

{/* Enhanced Test Button with Severity Tracking */}

{/* Enhanced Test Button with Severity Tracking */}
<button 
  className="inject-button"
  style={{ background: 'linear-gradient(135deg, #3B82F6 0%, #2563EB 100%)' }}
  onClick={async () => {
    console.log('🧪 Starting 10-attack test...\n');
    
    let results = {
      extreme: { detected: 0, missed: 0 },
      medium: { detected: 0, missed: 0 },
      subtle: { detected: 0, missed: 0 },
      total_tests: 10
    };
    
    for (let i = 0; i < 10; i++) {
      console.log(`\n═══ Test ${i+1}/10 ═══`);
      
      // Inject attack and wait for response
      const result = await handleInject();
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Use the returned result directly
      if (result && result.vae) {
        const error = result.vae.reconstruction_error;
        const isDetected = result.vae.attack_detected;
        
        // Classify severity based on reconstruction error
        let severity;
        if (error > 0.15) {
          severity = 'extreme';
          console.log(`   📊 Severity: EXTREME (error=${error.toFixed(4)})`);
        } else if (error > 0.08) {
          severity = 'medium';
          console.log(`   📊 Severity: MEDIUM (error=${error.toFixed(4)})`);
        } else {
          severity = 'subtle';
          console.log(`   📊 Severity: SUBTLE (error=${error.toFixed(4)})`);
        }
        
        // Track results
        if (isDetected) {
          results[severity].detected++;
          console.log(`   ✅ ${severity.toUpperCase()} attack DETECTED`);
        } else {
          results[severity].missed++;
          console.log(`   ⚠️ ${severity.toUpperCase()} attack EVADED detection`);
        }
      }
      
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    console.log('\n\n═══════════════════════════════════');
    console.log('📊 FINAL TEST SUMMARY');
    console.log('═══════════════════════════════════');
    console.log(`\n🔴 EXTREME Attacks:`);
    console.log(`   Detected: ${results.extreme.detected}`);
    console.log(`   Missed: ${results.extreme.missed}`);
    console.log(`   Detection Rate: ${results.extreme.detected + results.extreme.missed > 0 ? ((results.extreme.detected / (results.extreme.detected + results.extreme.missed)) * 100).toFixed(1) : 0}%`);
    
    console.log(`\n🟡 MEDIUM Attacks:`);
    console.log(`   Detected: ${results.medium.detected}`);
    console.log(`   Missed: ${results.medium.missed}`);
    console.log(`   Detection Rate: ${results.medium.detected + results.medium.missed > 0 ? ((results.medium.detected / (results.medium.detected + results.medium.missed)) * 100).toFixed(1) : 0}%`);
    
    console.log(`\n🟢 SUBTLE Attacks:`);
    console.log(`   Detected: ${results.subtle.detected}`);
    console.log(`   Missed: ${results.subtle.missed}`);
    console.log(`   Detection Rate: ${results.subtle.detected + results.subtle.missed > 0 ? ((results.subtle.detected / (results.subtle.detected + results.subtle.missed)) * 100).toFixed(1) : 0}%`);
    
    const totalDetected = results.extreme.detected + results.medium.detected + results.subtle.detected;
    const totalMissed = results.extreme.missed + results.medium.missed + results.subtle.missed;
    
    console.log(`\n🎯 OVERALL:`);
    console.log(`   Total Detected: ${totalDetected}/10`);
    console.log(`   Total Missed: ${totalMissed}/10`);
    console.log(`   Overall Detection Rate: ${((totalDetected / 10) * 100).toFixed(1)}%`);
    console.log('═══════════════════════════════════\n');
  }}
>
  <span className="button-icon">🧪</span>
  <span className="button-text">Run 10 Tests</span>
</button>

      {/* Attack Preview */}
      {selectedAttackData && (
        <div className="attack-preview">
          <h4 className="section-label">Attack Preview</h4>
          <div className="preview-content">
            <div className="preview-row">
              <span className="preview-label">Type:</span>
              <span className="preview-value">{selectedAttackData.name}</span>
            </div>
            <div className="preview-row">
              <span className="preview-label">Target:</span>
              <span className="preview-value">
                {stages.find(s => s.id === selectedStage)?.name}
              </span>
            </div>
            <div className="preview-row">
              <span className="preview-label">Impact:</span>
              <span className={`preview-value severity--${selectedAttackData.severity}`}>
                {selectedAttackData.severity.toUpperCase()}
              </span>
            </div>
            
            {/* Show last VAE result if available */}
            {lastResult && (
              <div className="preview-row">
                <span className="preview-label">Last Error:</span>
                <span className="preview-value">
                  {lastResult.reconstruction_error?.toFixed(4) || 'N/A'}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Recent Attacks History */}
      <div className="attack-history">
        <h4 className="section-label">Recent Attacks</h4>
        <div className="history-list">
          <div className="history-item">
            <span className="history-time">10:32:15</span>
            <span className="history-name">GenAI Attack #1</span>
            <span className="history-status history-status--mitigated">Mitigated</span>
          </div>
          <div className="history-item">
            <span className="history-time">10:28:42</span>
            <span className="history-name">Sensor Spoofing</span>
            <span className="history-status history-status--mitigated">Mitigated</span>
          </div>
          <div className="history-item">
            <span className="history-time">10:15:03</span>
            <span className="history-name">Overflow Attack</span>
            <span className="history-status history-status--mitigated">Mitigated</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AttackSelector;