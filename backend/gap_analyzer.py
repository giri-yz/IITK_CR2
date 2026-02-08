"""
Gap Analyzer - Convergence of GenAI and Digital Twin
Identifies cybersecurity gaps by combining generated attacks with twin simulations
"""
import numpy as np
from typing import List, Dict, Tuple
from attack_generator import AttackGenerator
from digital_twin import WaterTreatmentTwin
from vulnerability_db import VulnerabilityDB
import json


class GapAnalyzer:
    """
    Analyzes cybersecurity gaps by:
    1. Generating novel attacks with GenAI
    2. Simulating impacts with Digital Twin
    3. Identifying vulnerabilities from results
    """
    
    def __init__(self, attack_gen: AttackGenerator, twin: WaterTreatmentTwin, db: VulnerabilityDB):
        self.attack_gen = attack_gen
        self.twin = twin
        self.db = db
        
        print("✓ Gap Analyzer initialized")
    
    def discover_vulnerabilities(self, num_iterations: int = 200) -> Dict:
        """
        Main discovery loop: Generate attacks → Simulate → Analyze gaps
        """
        print(f"\n🔍 Starting vulnerability discovery ({num_iterations} iterations)...")
        
        # Generate comprehensive attack suite
        attack_suite = self.attack_gen.generate_attack_suite()
        
        all_simulations = []
        discovered_gaps = []
        
        for suite_idx, attack_scenario in enumerate(attack_suite):
            print(f"\n📦 Processing attack suite {suite_idx + 1}/{len(attack_suite)}: {attack_scenario['type']}")
            
            attack_samples = attack_scenario['samples']
            scenario_sims = []
            
            # Simulate each attack sample
            for i, attack_vector in enumerate(attack_samples[:50]):  # Limit per suite
                # Run digital twin simulation
                sim_result = self.twin.simulate_attack_impact(attack_vector, duration=150)
                
                # Store in database
                attack_id = self.db.add_generated_attack(
                    attack_type=attack_scenario['type'],
                    target=attack_scenario.get('targets', ['MULTIPLE'])[0] if 'targets' in attack_scenario else 'SYSTEM',
                    vector={'values': attack_vector.tolist()},
                    success_prob=1.0 if sim_result['process_failure'] else 0.0,
                    impact=sim_result['failure_mode']
                )
                
                # Store simulation result
                self.db.add_simulation_result(
                    attack_id=attack_id,
                    sim_type='PROCESS_IMPACT',
                    initial_state=sim_result['initial_state'],
                    attack_state={'attack_vector': attack_vector.tolist()},
                    final_state=sim_result['final_state'],
                    process_failure=sim_result['process_failure'],
                    failure_mode=sim_result['failure_mode'],
                    time_to_failure=sim_result['time_to_failure'] if sim_result['time_to_failure'] else 999.0,
                    affected=','.join(sim_result['affected_components'])
                )
                
                scenario_sims.append(sim_result)
                
                if (i + 1) % 10 == 0:
                    print(f"  Simulated {i + 1}/{len(attack_samples[:50])} attacks...")
            
            all_simulations.extend(scenario_sims)
            
            # Analyze this scenario for gaps
            scenario_gaps = self._analyze_scenario_gaps(attack_scenario, scenario_sims)
            discovered_gaps.extend(scenario_gaps)
        
        # Overall vulnerability analysis
        print("\n🧬 Analyzing system-wide vulnerabilities...")
        system_vulns = self.twin.identify_vulnerabilities(all_simulations)
        
        # Store system vulnerabilities
        for vuln in system_vulns:
            self.db.add_vulnerability(
                vuln_type=vuln['type'],
                severity=vuln['severity'],
                component=vuln.get('component', 'SYSTEM'),
                description=vuln['description'],
                impact_score=self._severity_to_score(vuln['severity']),
                confidence=0.85
            )
        
        # Generate process gaps
        process_gaps = self._identify_process_gaps(all_simulations)
        for gap in process_gaps:
            self.db.add_process_gap(**gap)
        
        # Generate mitigation strategies
        mitigations = self._generate_mitigations(system_vulns, process_gaps)
        
        print(f"\n✅ Discovery complete:")
        print(f"  Total simulations: {len(all_simulations)}")
        print(f"  Vulnerabilities found: {len(system_vulns)}")
        print(f"  Process gaps identified: {len(process_gaps)}")
        print(f"  Mitigation strategies: {len(mitigations)}")
        
        return {
            'total_simulations': len(all_simulations),
            'vulnerabilities': system_vulns,
            'process_gaps': process_gaps,
            'mitigations': mitigations,
            'success_rate': sum(1 for s in all_simulations if s['process_failure']) / len(all_simulations)
        }
    
    def _analyze_scenario_gaps(self, scenario: Dict, simulations: List[Dict]) -> List[Dict]:
        """Analyze gaps specific to an attack scenario"""
        gaps = []
        
        failures = [s for s in simulations if s['process_failure']]
        
        if len(failures) > len(simulations) * 0.3:  # More than 30% failure rate
            # This scenario reveals a significant gap
            gap = {
                'scenario_type': scenario['type'],
                'failure_rate': len(failures) / len(simulations),
                'severity': scenario['severity'],
                'common_failure_modes': self._get_common_failure_modes(failures),
                'description': f"{scenario['type']} attacks succeeded in {len(failures)}/{len(simulations)} cases"
            }
            gaps.append(gap)
        
        return gaps
    
    def _identify_process_gaps(self, simulations: List[Dict]) -> List[Dict]:
        """
        Identify specific process-level gaps from simulations
        """
        gaps = []
        
        # Gap 1: Rapid failure without detection window
        rapid_failures = [s for s in simulations if s['time_to_failure'] and s['time_to_failure'] < 20]
        if rapid_failures:
            avg_ttf = np.mean([s['time_to_failure'] for s in rapid_failures])
            gaps.append({
                'gap_type': 'INSUFFICIENT_DETECTION_TIME',
                'process_stage': 'OVERALL_SYSTEM',
                'min_thresh': 0.0,
                'max_thresh': avg_ttf,
                'blind_spot': avg_ttf,
                'missing': 'Early warning system for rapid anomalies',
                'recommendation': 'Implement predictive anomaly detection with <10s response time',
                'risk': 'CRITICAL'
            })
        
        # Gap 2: Tank overflow vulnerabilities
        overflow_failures = [s for s in simulations if s['failure_mode'] == 'TANK_OVERFLOW']
        if len(overflow_failures) > 10:
            gaps.append({
                'gap_type': 'INADEQUATE_OVERFLOW_PROTECTION',
                'process_stage': 'TANK_MANAGEMENT',
                'min_thresh': 0.8,
                'max_thresh': 1.0,
                'blind_spot': 15.0,
                'missing': 'Redundant level sensors and emergency shutoff',
                'recommendation': 'Install secondary level sensors with independent alarm system',
                'risk': 'HIGH'
            })
        
        # Gap 3: Pressure management
        pressure_failures = [s for s in simulations if 'PRESSURE' in s['failure_mode']]
        if pressure_failures:
            gaps.append({
                'gap_type': 'PRESSURE_CONTROL_WEAKNESS',
                'process_stage': 'PUMPING_SYSTEM',
                'min_thresh': 0.3,
                'max_thresh': 0.7,
                'blind_spot': 8.5,
                'missing': 'Pressure relief valves with automatic activation',
                'recommendation': 'Add pressure relief system with <5s activation time',
                'risk': 'HIGH'
            })
        
        # Gap 4: Coordinated attack detection
        all_components_affected = [s for s in simulations 
                                   if len(s['affected_components']) >= 3]
        if len(all_components_affected) > 20:
            gaps.append({
                'gap_type': 'MULTI_POINT_ATTACK_DETECTION',
                'process_stage': 'CROSS_SYSTEM',
                'min_thresh': 0.0,
                'max_thresh': 1.0,
                'blind_spot': 25.0,
                'missing': 'Correlation-based attack detection across multiple sensors',
                'recommendation': 'Implement multi-sensor correlation with graph-based anomaly detection',
                'risk': 'CRITICAL'
            })
        
        # Gap 5: Chemical dosing control
        conductivity_issues = [s for s in simulations if 'CONDUCTIVITY' in s['failure_mode']]
        if conductivity_issues:
            gaps.append({
                'gap_type': 'CHEMICAL_DOSING_VULNERABILITY',
                'process_stage': 'CHEMICAL_TREATMENT',
                'min_thresh': 0.4,
                'max_thresh': 0.6,
                'blind_spot': 12.0,
                'missing': 'Redundant conductivity measurement',
                'recommendation': 'Install backup AIT sensor with independent controller',
                'risk': 'MEDIUM'
            })
        
        return gaps
    
    def _generate_mitigations(self, vulnerabilities: List[Dict], 
                             process_gaps: List[Dict]) -> List[Dict]:
        """Generate mitigation strategies for discovered vulnerabilities"""
        mitigations = []
        
        # Store mitigations in database
        for vuln in vulnerabilities:
            if vuln['severity'] in ['CRITICAL', 'HIGH']:
                strategy = self._create_mitigation_strategy(vuln)
                
                # Get last vulnerability ID from DB
                vulns = self.db.get_vulnerabilities()
                if vulns:
                    vuln_id = vulns[0]['id']
                    
                    self.db.add_mitigation(
                        vuln_id=vuln_id,
                        strategy=strategy['name'],
                        complexity=strategy['complexity'],
                        effectiveness=strategy['effectiveness'],
                        time=strategy['deployment_time'],
                        description=strategy['description']
                    )
                    
                    mitigations.append(strategy)
        
        return mitigations
    
    def _create_mitigation_strategy(self, vulnerability: Dict) -> Dict:
        """Create mitigation strategy for a vulnerability"""
        vuln_type = vulnerability['type']
        
        strategies = {
            'RAPID_FAILURE': {
                'name': 'Predictive Early Warning System',
                'complexity': 'MEDIUM',
                'effectiveness': 0.85,
                'deployment_time': '2-3 weeks',
                'description': 'Deploy ML-based predictive model to detect anomalies 15-20s before failure'
            },
            'VULNERABLE_COMPONENT': {
                'name': 'Component Redundancy',
                'complexity': 'HIGH',
                'effectiveness': 0.90,
                'deployment_time': '4-6 weeks',
                'description': f"Install redundant sensor for {vulnerability.get('component', 'component')} with independent monitoring"
            },
            'RECURRING_FAILURE_MODE': {
                'name': 'Targeted Protection Mechanism',
                'complexity': 'MEDIUM',
                'effectiveness': 0.80,
                'deployment_time': '2-4 weeks',
                'description': f"Implement specific safeguard against {vulnerability.get('failure_mode', 'failure mode')}"
            }
        }
        
        return strategies.get(vuln_type, {
            'name': 'General Hardening',
            'complexity': 'LOW',
            'effectiveness': 0.60,
            'deployment_time': '1-2 weeks',
            'description': 'Implement general security hardening measures'
        })
    
    def _get_common_failure_modes(self, failures: List[Dict]) -> List[str]:
        """Get most common failure modes"""
        modes = {}
        for f in failures:
            mode = f['failure_mode']
            modes[mode] = modes.get(mode, 0) + 1
        
        return sorted(modes.items(), key=lambda x: x[1], reverse=True)[:3]
    
    def _severity_to_score(self, severity: str) -> float:
        """Convert severity to numerical score"""
        scores = {
            'LOW': 3.0,
            'MEDIUM': 5.5,
            'HIGH': 8.0,
            'CRITICAL': 9.5
        }
        return scores.get(severity, 5.0)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive vulnerability report from database"""
        vulnerabilities = self.db.get_vulnerabilities(mitigated=False)
        process_gaps = self.db.get_process_gaps()
        sim_stats = self.db.get_simulation_stats()
        mitigations = self.db.get_mitigation_summary()
        
        report = {
            'executive_summary': {
                'total_vulnerabilities': len(vulnerabilities),
                'critical_vulnerabilities': len([v for v in vulnerabilities if v['severity'] == 'CRITICAL']),
                'high_vulnerabilities': len([v for v in vulnerabilities if v['severity'] == 'HIGH']),
                'process_gaps': len(process_gaps),
                'simulations_run': sim_stats['total_simulations'],
                'process_failure_rate': sim_stats['process_failures'] / max(sim_stats['total_simulations'], 1)
            },
            'vulnerabilities': vulnerabilities,
            'process_gaps': process_gaps,
            'simulation_statistics': sim_stats,
            'mitigation_strategies': mitigations,
            'recommendations': self._generate_recommendations(vulnerabilities, process_gaps)
        }
        
        return report
    
    def _generate_recommendations(self, vulnerabilities: List[Dict], 
                                  gaps: List[Dict]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        critical_count = len([v for v in vulnerabilities if v['severity'] == 'CRITICAL'])
        if critical_count > 0:
            recommendations.append(
                f"URGENT: Address {critical_count} critical vulnerabilities immediately"
            )
        
        high_risk_gaps = [g for g in gaps if g['risk_level'] == 'CRITICAL']
        if high_risk_gaps:
            recommendations.append(
                f"Implement protection for {len(high_risk_gaps)} high-risk process gaps within 2 weeks"
            )
        
        recommendations.append(
            "Deploy continuous monitoring system with GenAI-based attack detection"
        )
        
        recommendations.append(
            "Conduct quarterly vulnerability assessments using GenTwin framework"
        )
        
        return recommendations


if __name__ == "__main__":
    print("Gap Analyzer - requires initialized AttackGenerator, DigitalTwin, and VulnerabilityDB")