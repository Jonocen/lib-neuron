CC      := gcc
AR      := ar
CFLAGS  := -Wall -Wextra -Wpedantic -std=c11 -Iinclude
RELEASE := -O2
DEBUG   := -O0 -g
PIC     := -fPIC

SRC := src/matrixcalculation.c \
	src/computervison.c \
	src/libvideo.c \
	src/dataset.c \
	src/layers.c \
       src/lossfunctions.c \
       src/optimizers.c \
       src/models.c

OBJ := matrixcalculation.o computervison.o libvideo.o dataset.o layers.o lossfunctions.o optimizers.o models.o
PIC_OBJ := matrixcalculation.pic.o computervison.pic.o libvideo.pic.o dataset.pic.o layers.pic.o lossfunctions.pic.o optimizers.pic.o models.pic.o
LIB := libneuron.a
SHARED_LIB := libneuron.so
EXAMPLE_BINARIES := $(patsubst examples/%.c,examples/%,$(wildcard examples/*.c))

.PHONY: all lib static shared debug clean examples sequential_xor_plugin Other_Exaple simple_compact

all: CFLAGS += $(RELEASE)
all: lib

lib: static shared

static: $(LIB)

shared: $(SHARED_LIB)

debug: CFLAGS += $(DEBUG)
debug: clean $(LIB)

$(LIB): $(OBJ)
	$(AR) rcs $@ $^

$(SHARED_LIB): $(PIC_OBJ)
	$(CC) -shared -o $@ $^ -lm

matrixcalculation.o: src/matrixcalculation.c include/matrixcalculation.h
	$(CC) $(CFLAGS) -c $< -o $@

computervison.o: src/computervison.c include/computervison.h
	$(CC) $(CFLAGS) -c $< -o $@

libvideo.o: src/libvideo.c include/libvideo.h include/computervison.h
	$(CC) $(CFLAGS) -c $< -o $@

dataset.o: src/dataset.c include/dataset.h include/computervison.h include/models.h
	$(CC) $(CFLAGS) -c $< -o $@

lossfunctions.o: src/lossfunctions.c include/lossfunctions.h
	$(CC) $(CFLAGS) -c $< -o $@

layers.o: src/layers.c include/layers.h include/matrixcalculation.h
	$(CC) $(CFLAGS) -c $< -o $@

optimizers.o: src/optimizers.c include/optimizers.h
	$(CC) $(CFLAGS) -c $< -o $@

models.o: src/models.c include/models.h include/matrixcalculation.h include/lossfunctions.h include/optimizers.h
	$(CC) $(CFLAGS) -c $< -o $@

matrixcalculation.pic.o: src/matrixcalculation.c include/matrixcalculation.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

computervison.pic.o: src/computervison.c include/computervison.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

dataset.pic.o: src/dataset.c include/dataset.h include/computervison.h include/models.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

libvideo.pic.o: src/libvideo.c include/libvideo.h include/computervison.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

lossfunctions.pic.o: src/lossfunctions.c include/lossfunctions.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

layers.pic.o: src/layers.c include/layers.h include/matrixcalculation.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

optimizers.pic.o: src/optimizers.c include/optimizers.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

models.pic.o: src/models.c include/models.h include/matrixcalculation.h include/lossfunctions.h include/optimizers.h
	$(CC) $(CFLAGS) $(RELEASE) $(PIC) -c $< -o $@

clean:
	rm -f $(OBJ) $(PIC_OBJ) $(LIB) $(SHARED_LIB) $(EXAMPLE_BINARIES)

examples: lib
	$(MAKE) -C examples

sequential_xor_plugin: lib
	$(MAKE) -C examples sequential_xor_plugin

Other_Exaple: lib
	$(MAKE) -C examples Other_Exaple

simple_compact: lib
	$(MAKE) -C examples simple_compact
