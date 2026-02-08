"""
Digital Twin Simulator for SWaT Water Treatment Process
Simulates process physics and attack impacts
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class ProcessState:
    """Represents the state of the water treatment process"""
    timestamp: float
    tank_levels: Dict[str, float]  # Tank levels (0-1 normalized)
    flow_rates: Dict[str, float]   # Flow rates (0-1 normalized)
    pump_states: Dict[str, bool]   # Pump on/off
    valve_states: Dict[str, float] # Valve positions (0-1)
    pressure: Dict[str, float]      # System pressures
    conductivity: float             # Chemical conductivity
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'tank_levels': self.tank_levels,
            'flow_rates': self.flow_rates,
            'pump_states': self.pump_states,
            'valve_states': self.valve_states,
            'pressure': self.pressure,
            'conductivity': self.conductivity
        }


class WaterTreatmentTwin:
    """
    Digital Twin of SWaT Water Treatment Process
    Simulates physical process and detects failures
    """
    
    def __init__(self):
        # Physical constraints (normalized to 0-1)
        self.TANK_CAPACITY = {
            'T101': 1.0,
            'T301': 1.0,
            'T401': 1.0,
            'T601': 1.0
        }
        
        self.SAFE_TANK_RANGE = {
            'T101': (0.2, 0.8),
            'T301': (0.3, 0.85),
            'T401': (0.25, 0.80),
            'T601': (0.2, 0.75)
        }
        
        self.PUMP_FLOW_RATE = {
            'P101': 0.15,
            'P201': 0.12,
            'P301': 0.18,
            'P401': 0.14
        }
        
        self.SAFE_PRESSURE_RANGE = (0.3, 0.7)
        self.SAFE_CONDUCTIVITY_RANGE = (0.4, 0.6)
        
        # Simulation parameters
        self.dt = 1.0  # Time step (seconds)
        
    def initialize_state(self) -> ProcessState:
        """Initialize process at safe operating point"""
        return ProcessState(
            timestamp=0.0,
            tank_levels={
                'T101': 0.5,
                'T301': 0.6,
                'T401': 0.55,
                'T601': 0.5
            },
            flow_rates={
                'F101': 0.15,
                'F201': 0.12,
                'F301': 0.14,
                'F401': 0.13
            },
            pump_states={
                'P101': True,
                'P201': True,
                'P301': True,
                'P401': True
            },
            valve_states={
                'MV101': 0.6,
                'MV201': 0.5,
                'MV301': 0.55,
                'MV401': 0.6
            },
            pressure={
                'P101': 0.5,
                'P201': 0.48,
                'P301': 0.52
            },
            conductivity=0.5
        )
    
    def simulate_step(self, state: ProcessState, control_input: Optional[Dict] = None) -> ProcessState:
        """
        Simulate one time step of the process
        """
        new_state = ProcessState(
            timestamp=state.timestamp + self.dt,
            tank_levels=state.tank_levels.copy(),
            flow_rates=state.flow_rates.copy(),
            pump_states=state.pump_states.copy(),
            valve_states=state.valve_states.copy(),
            pressure=state.pressure.copy(),
            conductivity=state.conductivity
        )
        
        # Apply control inputs if provided
        if control_input:
            if 'pump_states' in control_input:
                new_state.pump_states.update(control_input['pump_states'])
            if 'valve_states' in control_input:
                new_state.valve_states.update(control_input['valve_states'])
        
        # Simulate tank level dynamics (simplified mass balance)
        # T101: Raw water tank
        inflow_T101 = 0.20 if new_state.pump_states.get('P101', True) else 0.0
        outflow_T101 = new_state.flow_rates.get('F101', 0.15) * new_state.valve_states.get('MV101', 0.6)
        new_state.tank_levels['T101'] += (inflow_T101 - outflow_T101) * self.dt
        new_state.tank_levels['T101'] = np.clip(new_state.tank_levels['T101'], 0.0, 1.0)
        
        # T301: Chemical dosing tank
        inflow_T301 = outflow_T101 * 0.9  # Some loss
        outflow_T301 = new_state.flow_rates.get('F301', 0.14) * new_state.valve_states.get('MV301', 0.55)
        new_state.tank_levels['T301'] += (inflow_T301 - outflow_T301) * self.dt
        new_state.tank_levels['T301'] = np.clip(new_state.tank_levels['T301'], 0.0, 1.0)
        
        # T401: Filtration tank
        inflow_T401 = outflow_T301 * 0.95
        outflow_T401 = new_state.flow_rates.get('F401', 0.13) * new_state.valve_states.get('MV401', 0.6)
        new_state.tank_levels['T401'] += (inflow_T401 - outflow_T401) * self.dt
        new_state.tank_levels['T401'] = np.clip(new_state.tank_levels['T401'], 0.0, 1.0)
        
        # T601: Treated water tank
        inflow_T601 = outflow_T401 * 0.98
        outflow_T601 = 0.12  # Constant demand
        new_state.tank_levels['T601'] += (inflow_T601 - outflow_T601) * self.dt
        new_state.tank_levels['T601'] = np.clip(new_state.tank_levels['T601'], 0.0, 1.0)
        
        # Update flow rates based on pumps and valves
        new_state.flow_rates['F101'] = self.PUMP_FLOW_RATE['P101'] * new_state.valve_states['MV101'] if new_state.pump_states['P101'] else 0.0
        new_state.flow_rates['F301'] = self.PUMP_FLOW_RATE['P301'] * new_state.valve_states['MV301'] if new_state.pump_states['P301'] else 0.0
        new_state.flow_rates['F401'] = self.PUMP_FLOW_RATE['P401'] * new_state.valve_states['MV401'] if new_state.pump_states['P401'] else 0.0
        
        # Update pressure (simplified)
        new_state.pressure['P101'] = 0.5 + 0.2 * new_state.flow_rates['F101']
        new_state.pressure['P301'] = 0.48 + 0.15 * new_state.flow_rates['F301']
        
        # Conductivity dynamics (chemical dosing)
        new_state.conductivity = 0.5 + 0.1 * (new_state.tank_levels['T301'] - 0.5)
        
        return new_state
    
    def check_safety_violations(self, state: ProcessState) -> List[Dict]:
        """
        Check for safety violations in current state
        """
        violations = []
        
        # Check tank levels
        for tank, level in state.tank_levels.items():
            min_safe, max_safe = self.SAFE_TANK_RANGE.get(tank, (0.0, 1.0))
            
            if level < min_safe:
                violations.append({
                    'type': 'TANK_UNDERFLOW',
                    'component': tank,
                    'severity': 'HIGH' if level < min_safe * 0.5 else 'MEDIUM',
                    'value': level,
                    'threshold': min_safe,
                    'message': f'{tank} level {level:.3f} below safe minimum {min_safe:.3f}'
                })
            
            if level > max_safe:
                violations.append({
                    'type': 'TANK_OVERFLOW',
                    'component': tank,
                    'severity': 'CRITICAL' if level > max_safe * 1.2 else 'HIGH',
                    'value': level,
                    'threshold': max_safe,
                    'message': f'{tank} level {level:.3f} exceeds safe maximum {max_safe:.3f}'
                })
        
        # Check pressure
        for sensor, pressure in state.pressure.items():
            if pressure < self.SAFE_PRESSURE_RANGE[0]:
                violations.append({
                    'type': 'PRESSURE_LOW',
                    'component': sensor,
                    'severity': 'MEDIUM',
                    'value': pressure,
                    'threshold': self.SAFE_PRESSURE_RANGE[0],
                    'message': f'{sensor} pressure {pressure:.3f} below safe range'
                })
            
            if pressure > self.SAFE_PRESSURE_RANGE[1]:
                violations.append({
                    'type': 'PRESSURE_HIGH',
                    'component': sensor,
                    'severity': 'HIGH',
                    'value': pressure,
                    'threshold': self.SAFE_PRESSURE_RANGE[1],
                    'message': f'{sensor} pressure {pressure:.3f} exceeds safe range'
                })
        
        # Check conductivity
        if not (self.SAFE_CONDUCTIVITY_RANGE[0] <= state.conductivity <= self.SAFE_CONDUCTIVITY_RANGE[1]):
            violations.append({
                'type': 'CONDUCTIVITY_ABNORMAL',
                'component': 'AIT201',
                'severity': 'MEDIUM',
                'value': state.conductivity,
                'threshold': self.SAFE_CONDUCTIVITY_RANGE,
                'message': f'Conductivity {state.conductivity:.3f} outside safe range'
            })
        
        return violations
    
    def simulate_attack_impact(self, attack_vector: np.ndarray, 
                              duration: int = 100) -> Dict:
        """
        Simulate impact of attack vector on process
        
        Args:
            attack_vector: Sensor values during attack (normalized 0-1)
            duration: Simulation duration in time steps
        
        Returns:
            Simulation results with failure analysis
        """
        # Map attack vector to process inputs
        # Assuming attack_vector format: [FIT101, LIT101, FIT201, AIT201, FIT301, LIT301, 
        #                                  FIT401, LIT401, P101, P201, P301, P401, MV101, MV201]
        
        initial_state = self.initialize_state()
        current_state = initial_state
        
        states_history = [initial_state]
        violations_history = []
        
        process_failed = False
        failure_time = None
        failure_mode = None
        
        for t in range(duration):
            # Convert attack vector to control input
            control_input = self._attack_vector_to_control(attack_vector)
            
            # Simulate step
            current_state = self.simulate_step(current_state, control_input)
            states_history.append(current_state)
            
            # Check for violations
            violations = self.check_safety_violations(current_state)
            violations_history.append(violations)
            
            # Check for process failure
            if violations and not process_failed:
                critical_violations = [v for v in violations if v['severity'] in ['CRITICAL', 'HIGH']]
                if critical_violations:
                    process_failed = True
                    failure_time = t * self.dt
                    failure_mode = critical_violations[0]['type']
        
        # Analyze results
        affected_components = set()
        for violations in violations_history:
            for v in violations:
                affected_components.add(v['component'])
        
        return {
            'initial_state': initial_state.to_dict(),
            'final_state': current_state.to_dict(),
            'process_failure': process_failed,
            'failure_mode': failure_mode if process_failed else 'NO_FAILURE',
            'time_to_failure': failure_time if process_failed else None,
            'affected_components': list(affected_components),
            'total_violations': sum(len(v) for v in violations_history),
            'states_history': [s.to_dict() for s in states_history[::10]],  # Sample every 10 steps
            'violations_timeline': violations_history[::10]
        }
    
    def _attack_vector_to_control(self, attack_vector: np.ndarray) -> Dict:
        """Convert attack vector to control inputs"""
        # Map based on your feature order
        if len(attack_vector) >= 14:
            return {
                'pump_states': {
                    'P101': attack_vector[8] > 0.5,
                    'P201': attack_vector[9] > 0.5,
                    'P301': attack_vector[10] > 0.5,
                    'P401': attack_vector[11] > 0.5
                },
                'valve_states': {
                    'MV101': float(attack_vector[12]),
                    'MV201': float(attack_vector[13]) if len(attack_vector) > 13 else 0.5,
                    'MV301': 0.55,
                    'MV401': 0.6
                }
            }
        return {}
    
    def identify_vulnerabilities(self, simulation_results: List[Dict]) -> List[Dict]:
        """
        Analyze multiple simulation results to identify systemic vulnerabilities
        """
        vulnerabilities = []
        
        # Aggregate failure modes
        failure_modes = {}
        for sim in simulation_results:
            if sim['process_failure']:
                mode = sim['failure_mode']
                failure_modes[mode] = failure_modes.get(mode, 0) + 1
        
        # Identify most common failure points
        if failure_modes:
            most_common = max(failure_modes.items(), key=lambda x: x[1])
            vulnerabilities.append({
                'type': 'RECURRING_FAILURE_MODE',
                'failure_mode': most_common[0],
                'frequency': most_common[1],
                'severity': 'HIGH',
                'description': f'{most_common[0]} occurred in {most_common[1]} simulations'
            })
        
        # Identify components with most violations
        component_violations = {}
        for sim in simulation_results:
            for component in sim['affected_components']:
                component_violations[component] = component_violations.get(component, 0) + 1
        
        if component_violations:
            sorted_components = sorted(component_violations.items(), key=lambda x: x[1], reverse=True)
            for component, count in sorted_components[:3]:
                vulnerabilities.append({
                    'type': 'VULNERABLE_COMPONENT',
                    'component': component,
                    'violation_frequency': count,
                    'severity': 'HIGH' if count > len(simulation_results) * 0.5 else 'MEDIUM',
                    'description': f'{component} violated safety constraints in {count}/{len(simulation_results)} simulations'
                })
        
        # Check average time to failure
        ttf_values = [s['time_to_failure'] for s in simulation_results if s['time_to_failure']]
        if ttf_values:
            avg_ttf = np.mean(ttf_values)
            if avg_ttf < 30:  # Less than 30 seconds
                vulnerabilities.append({
                    'type': 'RAPID_FAILURE',
                    'avg_time_to_failure': avg_ttf,
                    'severity': 'CRITICAL',
                    'description': f'Average time to failure is only {avg_ttf:.1f}s - insufficient response time'
                })
        
        return vulnerabilities


if __name__ == "__main__":
    # Test simulation
    twin = WaterTreatmentTwin()
    
    # Normal operation
    state = twin.initialize_state()
    print("Initial state:")
    print(f"  Tank T101: {state.tank_levels['T101']:.3f}")
    
    # Simulate attack
    attack = np.array([0.9, 0.2, 0.8, 0.5, 0.7, 0.3, 0.6, 0.4, 1, 1, 0, 1, 0.9, 0.8])
    result = twin.simulate_attack_impact(attack, duration=100)
    
    print(f"\nSimulation result:")
    print(f"  Process failed: {result['process_failure']}")
    print(f"  Failure mode: {result['failure_mode']}")
    print(f"  Time to failure: {result['time_to_failure']}s")