To run program:
"distances_matrix.csv" must be in the same directory as the .py files.
Install pandas package.

source.py contains standalone program running Dijkstra's Algorithm.
    Run by using "py source.py" (Windows), "python3 source.py" (MacOS)

    This program will first ask for a starting point, you must enter a city and state combination in
    this format: "City, ST". State abbreviation must have both letters capitalized, and the city and
    state must be separated by a comma. Example: "Olympia, WA"

    The second prompt will ask for a destination city. Same formatting rules apply.
    To exit the program input "x" into either prompt.


AStar.py contains standalone program running A* Algorithm.
    Run by using "py AStar.py" (Windows), "python3 AStar.py" (MacOS)

    You will be prompted to input starting city and destination city. Same formatting rules apply
    here as the soure.py program.

test.py contains program that runs and times both algorithms. Compares funciton outputs to verify
    Run by using "py test.py" (Windows), "python3 test.py" (MacOS)

    The program will ask you how many times you want to run a test on the algorithms. The outputs
    will be the average run times of both algorithms. If the 2 algorithms had mismatching outputs
    the program will indicate which test # had the mismatch.

    To exit program input "x" during the prompt.