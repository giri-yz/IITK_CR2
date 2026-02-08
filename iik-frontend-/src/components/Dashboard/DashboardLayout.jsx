import React, { useState } from 'react';
import Header from './Header';
import PlantDiagram from '../WaterPlant/PlantDiagram';
import AttackSelector from '../AttackPanel/AttackSelector';
import ResponseLog from '../AILog/ResponseLog';
import TwinChart from '../Charts/TwinChart';
import GapPanel from '../GapSummary/GapPanel';
import AnomalyMeter from '../AnomalyMeter/AnomalyMeter';
import './DashboardLayout.css';

const DashboardLayout = () => {
  const [attackedStage, setAttackedStage] = useState(null);
  const [attackType, setAttackType] = useState(null);
  const [vaeResult, setVaeResult] = useState(null);
  const [attackPattern, setAttackPattern] = useState(null);  // NEW

const handleAttackInject = (attackData) => {
    console.log('=== ATTACK INJECTED ===');
    console.log('Attack Type:', attackData.attackType);
    console.log('Target Stage:', attackData.targetStage);
    console.log('VAE Result:', attackData.vaeResult);
    console.log('Sensor Values:', attackData.sensorValues);
    
    setAttackedStage(attackData.targetStage);
    setAttackType(attackData.attackType);
    setVaeResult(attackData.vaeResult);
    setAttackPattern(attackData.sensorValues);  // NEW
    
    // Auto-recovery after 8 seconds
    setTimeout(() => {
      console.log('=== AUTO RECOVERY ===');
      setAttackedStage(null);
      setAttackType(null);
      // Keep VAE result visible
    }, 8000);
  };

  return (
    <div className="dashboard">
      <Header systemStatus={vaeResult?.status} />
      <main className="dashboard-main">
        <div className="main-content">
          <div className="left-panel">
            <PlantDiagram 
              attackedStage={attackedStage}
              attackType={attackType}
            />
            
            <TwinChart 
              attackedStage={attackedStage}
              attackType={attackType}
            />
            
            <GapPanel 
              attackedStage={attackedStage}
              attackType={attackType}
              attackPattern={attackPattern}
            />
          </div>
          
          <div className="right-panel">
            <AttackSelector onAttackInject={handleAttackInject} />
            
            <AnomalyMeter vaeResult={vaeResult} />
            
            <ResponseLog 
              attackedStage={attackedStage}
              attackType={attackType}
            />
          </div>
        </div>
      </main>
    </div>
  );
};

export default DashboardLayout;