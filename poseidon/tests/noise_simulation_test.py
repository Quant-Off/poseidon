from poseidon.simulations.noise_modeling import BitFlipSimulation, PhaseFlipSimulation


bit_flip = BitFlipSimulation().simulate(verbose=2)
phase_flip = PhaseFlipSimulation().simulate(verbose=2)

print(f"결과: {bit_flip['fidelities'][5]:.6f}")
