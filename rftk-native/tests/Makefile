CXX=g++
CXXFLAGS=$(shell find ../ -maxdepth 1 -mindepth 1 -type d | sed -e 's~^~-I~')
LD=g++
LDFLAGS=$(shell find .. -name '*assert_util.o')
LDFLAGS+=$(shell find .. -name '*BufferCollection.o')

TEST_SRCS=$(shell ls test_*.cpp)
TESTS=$(TEST_SRCS:.cpp=)

default: all

all: $(TESTS)


.PHONY:clean

clean:
	rm $(TESTS)
