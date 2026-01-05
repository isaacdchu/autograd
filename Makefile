compile:
	g++-15 -Wall -Wextra -Wno-unused-parameter -std=c++23 main.cpp -o main.out
run:
	./main.out
clean:
	rm -f *.out
.PHONY: compile run clean
