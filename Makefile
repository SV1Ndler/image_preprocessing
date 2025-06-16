all:
	@cmake -DCMAKE_CXX_FLAGS=-pg -DCMAKE_EXE_LINKER_FLAGS=-pg -DCMAKE_SHARED_LINKER_FLAGS=-pg --preset="debug"
	@cmake --build --preset="debug"
	@ctest --preset debug
