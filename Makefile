compile:
	g++-15 -Wall -Wextra -Wno-unused-parameter -std=c++23 main.cpp -o main.out
run:
	./main.out
compile_tests:
	g++-15 -Wall -Wextra -Wno-unused-parameter -std=c++23 tests.cpp -o tests.out
run_tests:
	./tests.out
clean:
	rm -f *.out
.PHONY: compile run compile_tests run_tests clean
