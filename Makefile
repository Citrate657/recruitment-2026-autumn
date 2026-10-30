CXX_COMPLIER = g++
CXX_FLAGS = -std=c++17 -Wall -Wextra -O3 -march=native -fopenmp
LINKER = g++
LINKER_FLAGS = -fopenmp
INCLUDES = -I./include

# Default to the untouched reference implementation in src/. Select a numbered
# experiment without overwriting the baseline:
#   make EXPERIMENT=001_packed_fft_sparse_mel_simd
IMPLEMENTATION ?= src/melspec.cpp
EXPERIMENT ?=
ifneq ($(strip $(EXPERIMENT)),)
IMPLEMENTATION := exp/$(EXPERIMENT).cpp
endif

SRCS = src/main.cpp $(IMPLEMENTATION)
INCS = $(wildcard include/*.h)
OBJS = $(SRCS:.cpp=.o)
EXEC = melspec

.PHONY: all clean test format info

all: $(EXEC)

$(EXEC): $(OBJS)
	$(LINKER) $(LINKER_FLAGS) -o $@ $^

%.o: %.cpp $(INCS) Makefile
	$(CXX_COMPLIER) $(CXX_FLAGS) $(INCLUDES) -o $@ -c $<

clean:
	rm -f $(wildcard src/*.o exp/*.o) $(EXEC)

test: $(EXEC)
	@printf 'Implementation: %s\n' '$(IMPLEMENTATION)'
	python3 ./validate.py ./$(EXEC)

info:
	@printf 'Implementation: %s\n' '$(IMPLEMENTATION)'

format:
	clang-format -i $(wildcard src/*.cpp exp/*.cpp) $(INCS)
