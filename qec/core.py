
from ionq.qcircuitsim.qcircuit_simulator.qcircuit_simulator import QCircuitSimulator
from ionq.qcircuitsim.qcircuit_simulator.result import Result
import cirq
import numpy as np
from pathlib import Path
import stimcirq
from codes_constructions.syndrome_extraction_circuit_catalog import build_noiseless_syndrome_extraction_circuit, build_memory_experiment_circuit

from codes_constructions.stabilizer_code_factory import StabilizerCodeFactory

from codes_constructions.circuit_models import (
    SeqOperationConstraints,
    NoiseModel,
)
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import socket
from adaptive_scheduler import SlurmExecutor
from ionq.qcircuitsim.backends.quimb.quimb_mps_simulator import QuimbMPSSimulator
from ionq.qcircuitsim.backends.cirq.cirq_qcircuit_simulator import CirqSimulatorOptions, CirqQCircuitSimulator
from ionq.qcircuitsim.backends.cirq.cirq_circuit import CirqCircuit
from ionq.qcircuitsim.qcircuit_simulator.qcircuit_simulator_options import QCircuitSimulatorOptions
import sinter
import stim
import time
from pipefunc.resources import Resources
from pipefunc import PipeFunc, Pipeline
from pipefunc.typing import Array
from itertools import chain
import json
from ast import literal_eval


def map_result_to_stim_output(result: Result):
    """Map the result of a QCircuitSimulator simulation to the output of a stim simulation.
    Args:
        result (Result): The result of a QCircuitSimulator simulation."""
    measurements = result.measurements
    keys = sorted([key for key in result.measurements.keys() if key.isnumeric()], key = int)
    num_shots = result.num_shots
    num_measures = len(keys)
    stim_like_output = np.zeros((num_shots,num_measures),dtype=np.bool_)
    for i, key in enumerate(keys):
        stim_like_output[:,i] = np.bool_(np.array([measurements[key].reshape(-1)]))
    return stim_like_output

def build_noiseless_code_circuit(code,constraints):
    nm = IonChainNoiseModel(p_CNOT=0.001, meas_slower_factor=30)
    circuit = build_memory_experiment_circuit(code, constraints, nm, "z", 3)

    stripped_circuit = stim.Circuit()
    for instruction in circuit:
        if instruction.name == "DEPOLARIZE1":
            continue
        if instruction.name == "DEPOLARIZE2":
            continue
        if instruction.name == "DETECTOR":
            continue
        if instruction.name == "OBSERVABLE_INCLUDE":
            continue
        if instruction.name == "M":
            stripped_circuit.append("M", instruction.targets_copy())
            continue
        stripped_circuit.append(instruction)
    return stripped_circuit

class IonChainNoiseModel(NoiseModel):
    def __init__(self, p_CNOT: float, meas_slower_factor: float):
        self._meas_slower_factor = meas_slower_factor
        super().__init__(p_CNOT, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01 * meas_slower_factor)

    @property
    def meas_slower_factor(self) -> float:
        return self._meas_slower_factor