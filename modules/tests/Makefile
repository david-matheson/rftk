CXX=g++
CXXFLAGS=$(shell find ../ -maxdepth 2 -mindepth 1 -type d | sed -e 's~^~-I~')
LD=g++
LDFLAGS=-L../../build/debug/buffers/native -L../../build/debug/asserts/native
LDFLAGS+=-l:BufferCollection.os -l:asserts.os

TEST_SRCS=$(shell ls test_*.cpp)
TESTS=$(TEST_SRCS:.cpp=)

default: all

all: $(TESTS)


.PHONY:clean

clean:
	rm $(TESTS)
