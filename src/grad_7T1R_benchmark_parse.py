from optparse import OptionParser
from sys import argv

with open(argv[1],'r') as filein:
    lines=filein.readlines()

parser=OptionParser()
parser.add_option("-d", dest="data_set", help="input datset name", metavar="input_set")
parser.add_option("-l", dest='layer_desc', help="layer description", metavar="layers")
parser.add_option("-b", dest='batch_size', help="batch size", metavar="batch_size", type="int")
parser.add_option("-e", dest='epoch', help="number of epochs", metavar="epoch", type="int")
parser.add_option("--first", dest='data_first', help="first N data points", metavar="data_first", type="int")
parser.add_option('-s', dest='synapse_type', help='synapse type', metavar='synapse_type')
parser.add_option('--read-time', dest='read_time', help="synapse read time", metavar='read_time', type='float')
parser.add_option('--neuron-power', dest='p_neuron', help="neuron power", metavar='p_neuron', type='float')
parser.add_option('--notest', dest='notest', help="skip test", metavar='notest')
parser.add_option('--seed', dest='seed', help='np random seed', metavar='seed', type='int')
#parser.add_option('--std-step', dest='standard_step', help='use standard time step (shorter)', metavar='standard_step', type='int')
parser.add_option('--stdp-step', dest='stdp_step', help='stdp steps', metavar='stdp_step', type='int')
parser.add_option('--scale', dest='scale', help='layer size scaled by (unused)', metavar='scale', type='float')

results={}
k2_all={}
for line in lines:
    if '|' in line and line.startswith('src/grad_7T1R_benchmark.py'):
        options, args = parser.parse_args(args=line.split('|')[0].split())
        # if options.seed==15786 and options.standard_step==0: # code in this batch parsed stdstep as str
        #     continue
        if options.seed not in results:
            results[options.seed]={}
        if options.seed not in k2_all:
            k2_all[options.seed]=set()
        k1=(options.scale, options.data_set, options.stdp_step, options.layer_desc.replace(',','-'))
        k2=(options.synapse_type,)
        k2_all[options.seed].add(k2)
        if k1 not in results[options.seed]:
            results[options.seed][k1]={}
        if k2 in results[options.seed][k1]:
            print('duplicate:', options.seed, k1, k2, results[options.seed][k1][k2], list(map(float,line.split('|')[1].split())))
        results[options.seed][k1][k2]=list(map(float,line.split('|')[1].split()))

sep=','
fout=open(argv[1]+'.csv','w')
for seed in results.keys():
    print('seed', seed, file=fout)
    k2_list=sorted(k2_all[seed])
    print(*['','','',''], sep=sep, end=sep, file=fout)
    for k2 in k2_list:
        print(','.join(map(str,k2)), *['']*6, sep=sep, end=sep, file=fout)
    print(file=fout)
    print(*['scale','dataset','tau','layers'], sep=sep, end=sep, file=fout)
    for k2 in k2_list:
        print(*['read','reverse','update','pre','post','neuron','total'], sep=sep, end=sep, file=fout)
    print(file=fout)
    for k1 in sorted(results[seed].keys()):
        valdict=results[seed][k1]
        print(*k1, sep=sep, end=sep, file=fout)
        for k2 in k2_list:
            print(*valdict.get(k2,['#N/A']*7), sep=sep, end=sep, file=fout)
        print(file=fout)
