cc -c -fPIC tsintegrate.c -o tsintegrate.o
cc -shared -Wl,-soname,tsintegrate.so -o tsintegrate.so tsintegrate.o
rm tsintegrate.o
