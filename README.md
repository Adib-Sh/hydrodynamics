# Smoothed Particle Hydrodynamics (SPH) Simulation

A Python implementation of the Smoothed Particle Hydrodynamics method for simulating fluid dynamics and planetary collisions.

## Overview

This project implements SPH to solve two main problems:
- **1-D Sod's Shock Tube**: A classic test case for compressible fluid dynamics
- **3-D Planetary Collisions**: Simulation of gravitational interactions and collisions between planetary bodies

## Features

- Lagrangian particle-based approach for fluid dynamics
- Cubic spline smoothing kernel implementation
- Artificial viscosity for shock wave modeling
- Gravitational force calculations with softening
- NumPy broadcasting for efficient pairwise particle interactions
- Time integration using SciPy's `solve_ivp`

## Requirements

- Python 3.x
- NumPy
- SciPy
- Matplotlib (for visualization)

## Usage

### 1-D Shock Tube
The shock tube simulation models gas dynamics in a tube with initial high and low pressure regions separated by a membrane. The simulation runs from t=0 when the membrane is removed.

### 3-D Planetary Collision
The planetary collision simulation loads particle data from `Planet600.dat` and creates a two-planet system with:
- Spatial offsets between planets
- Individual rotational velocities
- Gravitational interactions with softening
- Short-range and long-range force calculations

## Results

- The 1-D shock tube results show good agreement with analytical solutions
- Animations and visualization outputs are available in the project repository
- The 3-D simulations capture collision dynamics and potential satellite formation

## Repository

Full code and animations available at the Gitlab repository.

## References

Implementation based on standard SPH methods and follows the computational approach described in the project lab manual.