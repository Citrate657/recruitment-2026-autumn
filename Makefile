CXX_COMPILER = g++
CXX_FLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -fopenmp
LINKER = g++
LINKER_FLAGS = -fopenmp
INCLUDES = -I./include

FINAL_SINGLE = 008_single_node_cpu_pipelined
FINAL_THREE_NODE = 009_three_node_cpu_pipelined
EXPERIMENT ?=

ifneq ($(filter $(EXPERIMENT),$(FINAL_SINGLE) $(FINAL_THREE_NODE)),)
$(error Use 'make final-single' or 'make final-three-node' for final candidates)
endif

ifeq ($(strip $(EXPERIMENT)),)
IMPLEMENTATION := exp/$(FINAL_SINGLE).cpp
SRCS := $(IMPLEMENTATION)
else
IMPLEMENTATION := exp/$(EXPERIMENT).cpp
SRCS := src/main.cpp $(IMPLEMENTATION)
endif

INCS = $(wildcard include/*.h)
OBJS = $(SRCS:.cpp=.o)
EXEC = melspec
FINAL_THREE_NODE_EXEC = melspec-final-three-node
BASELINE_EXEC = melspec-baseline

.PHONY: all baseline clean final-single final-three-node format info test

all: final-single

$(EXEC): $(OBJS)
	$(LINKER) $(LINKER_FLAGS) -o $@ $^

%.o: %.cpp $(INCS) Makefile
	$(CXX_COMPILER) $(CXX_FLAGS) $(INCLUDES) -o $@ -c $<

baseline:
	$(CXX_COMPILER) $(CXX_FLAGS) $(INCLUDES) src/main.cpp src/melspec.cpp \
		-o $(BASELINE_EXEC)

final-single: $(EXEC)

final-three-node: exp/$(FINAL_THREE_NODE).cpp $(INCS)
	$(CXX_COMPILER) $(CXX_FLAGS) $(INCLUDES) \
		exp/$(FINAL_THREE_NODE).cpp -o $(FINAL_THREE_NODE_EXEC)

clean:
	rm -f $(wildcard src/*.o exp/*.o) \
		$(EXEC) $(FINAL_THREE_NODE_EXEC) $(BASELINE_EXEC)

test: $(EXEC)
	@printf 'Implementation: %s\n' '$(IMPLEMENTATION)'
	python3 ./validate.py ./$(EXEC)

info:
	@printf 'Implementation: %s\n' '$(IMPLEMENTATION)'
	@printf 'Final three-node executable: ./%s\n' '$(FINAL_THREE_NODE_EXEC)'

format:
	clang-format -i $(wildcard src/*.cpp exp/*.cpp exp/*.cu) $(INCS)
