import React, { useState, useEffect, useRef } from 'react';
import './ResponseLog.css';

const ResponseLog = ({ attackedStage, attackType }) => {
  const [logs, setLogs] = useState([
    {
      id: 'initial-1',
      timestamp: '10:32:15',
      type: 'info',
      message: '✓ System monitoring active',
      stage: null
    },
    {
      id: 'initial-2',
      timestamp: '10:28:42',
      type: 'success',
      message: '✓ Previous attack mitigated successfully',
      stage: 'stage3'
    }
  ]);
  
  const logContainerRef = useRef(null);
  const timeoutsRef = useRef([]); // Track timeouts for cleanup

  // Auto-scroll to bottom when new logs appear
  useEffect(() => {
    if (logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs]);

  // Add logs when attack is injected
  useEffect(() => {
    if (attackedStage && attackType) {
      // Clear any existing timeouts to prevent duplicates
      timeoutsRef.current.forEach(timeout => clearTimeout(timeout));
      timeoutsRef.current = [];

      const attackId = Date.now(); // Unique ID for this attack sequence
      const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });
      
      // Create scheduled logs
      const scheduledLogs = [
        { delay: 500, data: {
          timestamp,
          type: 'warning',
          message: `⚠️ Anomaly detected at ${getStageName(attackedStage)}`,
          stage: attackedStage
        }},
        { delay: 1500, data: {
          timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
          type: 'info',
          message: `🔍 Digital twin analyzing deviation...`,
          stage: attackedStage
        }},
        { delay: 2500, data: {
          timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
          type: 'danger',
          message: `🚨 Attack identified: ${getAttackName(attackType)}`,
          stage: attackedStage
        }},
        { delay: 3500, data: {
          timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
          type: 'processing',
          message: `🟣 AI mitigation initiated`,
          stage: attackedStage
        }},
        { delay: 4500, data: {
          timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
          type: 'info',
          message: `⚙️ Adjusting valve MV${attackedStage.slice(-1)}01 to 45%`,
          stage: attackedStage
        }},
        { delay: 5500, data: {
          timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
          type: 'info',
          message: `⚙️ Reducing pump P${attackedStage.slice(-1)}01 flow rate`,
          stage: attackedStage
        }},
        { delay: 6500, data: {
          timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
          type: 'success',
          message: `✅ Mitigation successful - System stabilizing`,
          stage: attackedStage
        }},
        { delay: 7500, data: {
          timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
          type: 'warning',
          message: `📋 Cybersecurity gap identified and logged`,
          stage: attackedStage
        }},
        { delay: 8000, data: {
          timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
          type: 'success',
          message: `✓ System restored to normal operation`,
          stage: attackedStage
        }}
      ];

      // Schedule all logs with unique IDs
      scheduledLogs.forEach((log, index) => {
        const timeoutId = setTimeout(() => {
          addLog({
            ...log.data,
            id: `attack-${attackId}-step-${index}` // Guaranteed unique
          });
        }, log.delay);
        
        timeoutsRef.current.push(timeoutId);
      });
    }

    // Cleanup on unmount
    return () => {
      timeoutsRef.current.forEach(timeout => clearTimeout(timeout));
    };
  }, [attackedStage, attackType]);

  const addLog = (logData) => {
    setLogs(prev => [...prev, logData]);
  };

  const getStageName = (stageId) => {
    const stageNames = {
      stage1: 'Raw Water Intake',
      stage2: 'Chemical Dosing',
      stage3: 'Filtration',
      stage4: 'UV Treatment',
      stage5: 'pH Adjustment',
      stage6: 'Clean Water Storage'
    };
    return stageNames[stageId] || stageId;
  };

  const getAttackName = (type) => {
    const attackNames = {
      sensor_spoofing: 'Sensor Spoofing',
      plc_manipulation: 'PLC Manipulation',
      genai_attack_1: 'GenAI Attack Pattern #1',
      overflow_attack: 'Overflow Attack',
      contamination: 'Water Contamination',
      denial_of_service: 'Denial of Service'
    };
    return attackNames[type] || type;
  };

  const clearLogs = () => {
    // Clear all pending timeouts
    timeoutsRef.current.forEach(timeout => clearTimeout(timeout));
    timeoutsRef.current = [];
    
    setLogs([{
      id: `clear-${Date.now()}`,
      timestamp: new Date().toLocaleTimeString('en-US', { hour12: false }),
      type: 'info',
      message: '🔄 Logs cleared',
      stage: null
    }]);
  };

  return (
    <div className="response-log">
      <div className="log-header">
        <div className="log-title">
          <h3>AI Response Log</h3>
          <span className="log-count">{logs.length} entries</span>
        </div>
        <button className="clear-button" onClick={clearLogs} title="Clear logs">
          🗑️
        </button>
      </div>

      <div className="log-container" ref={logContainerRef}>
        {logs.map(log => (
          <div 
            key={log.id} 
            className={`log-entry log-entry--${log.type}`}
          >
            <span className="log-timestamp">{log.timestamp}</span>
            <span className="log-message">{log.message}</span>
          </div>
        ))}
      </div>

      <div className="log-footer">
        <div className="log-status">
          <span className="status-dot status-dot--active"></span>
          <span className="status-text">Live Monitoring</span>
        </div>
      </div>
    </div>
  );
};

export default ResponseLog;