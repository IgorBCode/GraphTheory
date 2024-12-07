# Setup:
- Install pandas package.
> [!IMPORTANT]
> For the program to work, "distances_matrix.csv" must be in the same directory as the .py files.

# source.py
Contains standalone program running Dijkstra's Algorithm.

## Run Instructions
Run by using "py source.py" (Windows), "python3 source.py" (MacOS)

## Operation
This program will first ask for a starting point, you must enter a city and state combination in
this format: "City, ST". State abbreviation must have both letters capitalized, and the city and
state must be separated by a comma. Example: "Olympia, WA"

## Exit
The second prompt will ask for a destination city. Same formatting rules apply.
To exit the program input "x" into either prompt.


# AStar.py 
Contains standalone program running A* Algorithm.

## Run Instructions
Run by using "py AStar.py" (Windows), "python3 AStar.py" (MacOS)

## Operation
You will be prompted to input starting city and destination city. Same formatting rules apply
here as the soure.py program.

# test.py 
Contains program that runs and times both algorithms. Compares funciton outputs to verify

## Run Instructions
Run by using "py test.py" (Windows), "python3 test.py" (MacOS)

## Operation
The program will ask you how many times you want to run a test on the algorithms. The outputs
will be the average run times of both algorithms. If the 2 algorithms had mismatching outputs
the program will indicate which test # had the mismatch.

## Exit
To exit program input "x" during the prompt.
