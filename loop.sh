set +e

ret=1
while [ $ret -gt 0 ]
do
time python3 src/stdp_conv.py
ret=$?
done
