CXX=g++
CXXFLAGS=$(shell find ../ -maxdepth 1 -mindepth 1 -type d | sed -e 's~^~-I~')
LD=g++
LDFLAGS=-L../../build/debug/buffers -L../../build/debug/assert_util
LDFLAGS+=-l:BufferCollection.os -l:assert_util.os

TEST_SRCS=$(shell ls test_*.cpp)
TESTS=$(TEST_SRCS:.cpp=)

default: all

all: $(TESTS)


.PHONY:clean

clean:
	rm $(TESTS)
