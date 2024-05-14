import sys
import matplotlib
import matplotlib.figure
import matplotlib.backends.backend_svg
import numpy as np
import os

Ntrain=60000
Ntest=10000
Tforward=1e-9

listfile=open(sys.argv[1], 'r')

names=[]
types=[]
batch_sizes=[]
acc=[]
energy=[]
Nin=[]
Nout=[]

for line in listfile:
    line=line.strip()
    if len(line)==0:
        continue
    fields = line.split(' ')
    name, Ne, train_batch_size, epoch, syn_type = fields[:5]
    syn_params = fields[5:]
    train_batch_size=int(train_batch_size)
    epoch=int(epoch)
    names.append(name)
    types.append(syn_type)
    batch_sizes.append(train_batch_size)
    datafile=open(os.path.join('outputs', '_'.join(fields[1:])), 'r')
    acc.append([])
    energy.append([])
    Nin.append([])
    Nout.append([])
    for current_batch_size in [train_batch_size for _ in range(epoch*(Ntrain//train_batch_size))]+[Ntest]:
        correct_cnt=0
        energy_tot=0
        in_tot=0
        out_tot=0
        for sampleid in range(current_batch_size):
            try:
                datafields = list(map(float, datafile.readline().split()))
                is_correct, power_forward, energy_update, count_input_spike, count_output_spike = datafields[:5]
            except ValueError:
                is_correct, power_forward, energy_update, count_input_spike, count_output_spike = [0., 0., 0., 0., 0.]
            correct_cnt+=is_correct
            energy_tot+=power_forward*Tforward+energy_update
            in_tot+=count_input_spike
            out_tot+=count_output_spike
        acc[-1].append(correct_cnt/current_batch_size)
        energy[-1].append(energy_tot/current_batch_size)
        Nin[-1].append(in_tot/current_batch_size)
        Nout[-1].append(out_tot/current_batch_size)

fig=matplotlib.figure.Figure(figsize=(8, 6), dpi=400, tight_layout=True)
matplotlib.backends.backend_svg.FigureCanvas(fig)
figplot=fig.subplots()
figplot.set_xlabel('Number of training samples')
figplot.set_ylabel('Accuracy (%)')
figplotr = figplot.twinx()
figplotr.set_ylabel('Resistive energy(J/image)')
for fileid in range(len(names)):
    if names[fileid].startswith('software'):
        mark='+'
    elif names[fileid].startswith('device-c'):
        mark='x'
    elif names[fileid].startswith('device-p'):
        mark='*'
    elif names[fileid].startswith('vteam-'):
        mark='o'
    accline=figplot.plot(batch_sizes[fileid]*np.arange(1, len(acc[fileid])), acc[fileid][1:], mark+'-', label=names[fileid])[0]
    if types[fileid]!='ideal':
        figplotr.plot(batch_sizes[fileid]*np.arange(len(energy[fileid])), energy[fileid], mark+'--', color=accline.get_color())
box = figplot.get_position()
# legendfill=0.4
# figplot.set_position([box.x0, box.y0+box.height*legendfill, box.width, box.height*(1-legendfill)])
figplot.legend(ncol=4, bbox_to_anchor=[0, -0.15], loc='upper left')
fig.savefig('acc_energy_time.png', format='png')

fig=matplotlib.figure.Figure(figsize=(8, 6), dpi=400, tight_layout=True)
matplotlib.backends.backend_svg.FigureCanvas(fig)
figplot=fig.subplots()
figplot.set_xlabel('Resistive energy(J/image)')
figplot.set_ylabel('Accuracy (%)')
figplot.set_xscale('log', base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9])
groups=['software', 'device-c', 'device-p', 'vteam']
acclists={gr:[] for gr in groups}
energylists={gr:[] for gr in groups}
for fileid in range(len(names)):
    batchid=-1# np.argmax(acc[fileid])
    if batchid!=-1:
        import warnings
        warnings.warn(RuntimeWarning('batchid should be -1'))
    try:
        del gr
    except:
        pass
    if names[fileid].startswith('software'):
        gr='software'
    elif names[fileid].startswith('device-c'):
        gr='device-c'
    elif names[fileid].startswith('device-p'):
        gr='device-p'
    elif names[fileid].startswith('vteam-'):
        gr='vteam'
    acclists[gr].append(acc[fileid][batchid])
    energylists[gr].append(energy[fileid][batchid])
for gr in groups:
    if gr=='software':
        energy_min=np.min([np.min(l) for name,l in energylists.items() if name!=gr])
        energy_max=np.max([np.max(l) for name,l in energylists.items() if name!=gr])
        figplot.plot([energy_min, energy_max], [acclists[gr][0], acclists[gr][-1]], '-', label=gr)
    else:
        figplot.plot(energylists[gr], acclists[gr], '-x', label=gr)
figplot.legend()
fig.savefig('acc_energy.png', format='png')
