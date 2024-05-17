#!/bin/bash

# Define the number of iterations you want to run
NUM_ITERATIONS=2

# Loop to run the Python script multiple times
for ((i=1; i<=$NUM_ITERATIONS; i++)); do
    echo "Iteration $i"
    python main.py  # Replace 'main.py' with the name of your Python script
done
