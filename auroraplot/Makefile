CC=gcc
UNAME=$(shell uname -s)                                                      
ifeq ($(UNAME),Darwin)
    CFLAGS=-Wl,-install_name,tsintegrate.so
else                                                                            
    CFLAGS=-Wl,-soname,tsintegrate.so
endif

tsintegrate.so: tsintegrate.o
	$(CC) -shared $(CFLAGS) -o tsintegrate.so tsintegrate.o

tsintegrate.o:
	$(CC) -c -fPIC tsintegrate.c -o tsintegrate.o

.PHONY: clean
clean:
	rm tsintegrate.o tsintegrate.so
