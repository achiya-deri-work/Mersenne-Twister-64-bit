# Mersenne-Twister-64-bit
A simple random or numpy.random like module implementing Mersenne Twister 64-bit and offering some basic random methods.

# Mersenne Twister 64-bit PRNG Python Module

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Introduction

This Python module implements the Mersenne Twister 64-bit Pseudo-Random Number Generator (PRNG). The Mersenne Twister is a widely-used PRNG known for its long period and good statistical properties, making it suitable for various randomization tasks.

The module provides a user-friendly interface for generating random numbers, shuffling sequences, and sampling elements. It also includes methods for seeding the generator, ensuring reproducibility, and verifying the correctness of the generator's implementation through testing.

## Features

- Generate random floating-point numbers and integers within specified ranges.
- Shuffle sequences using the Mersenne Twister PRNG.
- Sample elements from sequences with optional uniqueness.
- Set and update the internal state of the generator using seed values.
- Verify generator correctness through predefined tests.
- Access to the Mersenne Twister generator methods and attributes at the module level.

## Usage

1. Clone the repository:
   ```bash
   gh repo clone achiya-deri-work/Mersenne-Twister-64-bit
   cd Mersenne-Twister-64-bit

2. import the mt module to use various methods:
   ```bash
   import mt

3. Use the module's methods for various randomization tasks:
   ```bash
   random_value = mt.random()
   random_int = mt.randint(1, 100)
   shuffled_list = mt.shuffle([1, 2, 3, 4, 5])
   sampled_elements = mt.sample([1, 2, 3, 4, 5], k=2)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Credits
This module is developed by Achiya Deri.

## Feedback and Contributions
Feedback, suggestions, and contributions are welcome! Feel free to open issues or pull requests.
