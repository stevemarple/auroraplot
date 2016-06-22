cc -c -fPIC tsintegrate.c -o tsintegrate.o
unamestr=`uname`
if [[ "$unamestr" == 'Darwin' ]]; then
    cc -shared -Wl,-install_name,tsintegrate.so -o tsintegrate.so tsintegrate.o
else
    cc -shared -Wl,-soname,tsintegrate.so -o tsintegrate.so tsintegrate.o
fi
rm tsintegrate.o
