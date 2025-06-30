from neuron import h, gui
from neuron.units import ms, mV
import time as clock
import os
import re
import numpy
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import random
import math
import csv
import pandas as pd
import gc
from tkinter import Tk
import tkinter.filedialog as fd
from pathlib import Path

h.load_file("stdrun.hoc")


#### Following functions set up the model, when you import the LPLC2 model.
def instantiate_swc(filename):
    ''' 
    Load swc file and instantiate it as cell
    Code source: https://www.neuron.yale.edu/phpBB/viewtopic.php?t=3257
    '''

    # load helper library, included with Neuron
    h.load_file('import3d.hoc')

    # load data
    cell = h.Import3d_SWC_read()
    cell.input(filename)

    # # instantiate
    i3d = h.Import3d_GUI(cell,0)
    i3d.instantiate(None)

def change_Ra(ra=41.0265, electrodeSec=None, electrodeVal=None):
    for sec in h.allsec():
        sec.Ra = ra
    if electrodeSec is not None:
        electrodeSec.Ra = electrodeVal

def change_gLeak(gleak=0.000239761, erev=-58.75, electrodeSec=None, electrodeVal=None):
    for sec in h.allsec():
        sec.insert('pas')
        for seg in sec:
            seg.pas.g = gleak
            seg.pas.e = erev
    if electrodeSec is not None:
        for seg in electrodeSec:
            seg.pas.g = electrodeVal
            seg.pas.e = 0

def change_memCap(memcap=1.3765, electrodeSec=None, electrodeVal=None):
    for sec in h.allsec():
        sec.cm = memcap
    if electrodeSec is not None:
        electrodeSec.cm = electrodeVal

def nsegDiscretization(sectionListToDiscretize):
    #this function iterates over every section, calculates its spatial constant (lambda), and checks if the length of the segments within this section are less than 1/10  lambda
    #if true, nothing happens
    #if false, the section is broken up until into additional segments until the largest segment is no greater in length than 1/10 lambda

    #NOTE TO SELF, NOT IDEAL BC ONLY INCREASES, DOESNT DECREASE, WRITE CODE TO DECREMENT SEGMENT NUMBER IF APPLICABLE
    #NOTE need a check on if num seg > max allowaable num seg to avoid error
    #TODO FIX TO SAVE ON COMPUTATIONAL COMPLEXITY

    for sec in sectionListToDiscretize:
        #old code which may be useful for debugging
        #secMorphDict = sec.psection()['morphology']

        #by calling sec.psection, we can return the density mechanisms for the given section in a dictionary
        #this lets us access the gleak value for a given section, which we use to calculate the membrane resistance
        secDensityMechDict = sec.psection()['density_mechs']

        #using the membrane resistance, section diameter, and section axial resistance, we calculate the spatial constant (lambda) for a given section
        secLambda = math.sqrt( ( (1 / secDensityMechDict['pas']['g'][0]) * sec.diam) / (4*sec.Ra) )

        #debugging print statement
        #print("\nlength of section", sec, "is", (sec.L/sec.nseg)/secLambda, "lambda | sec.L:", sec.L, "| sec.nseg:", sec.nseg, "| lambda:", secLambda)

        #if the segment length of a section is greater than 1/10 lambda, then the section is broken into additional segments
        #the segment count per section must be odd to avoid issues with the way NEURON handles midpoints for internal calculations, so if the necessary number of segments to reach 
        #a max length of 1/10 lambda is even, an extra section is added
        if (sec.L/sec.nseg)/secLambda > 0.1:
            #debugging print statement
            #print("\nlength of section", sec, "is", (sec.L/sec.nseg)/secLambda, "lambda")

            numSeg_log2 = math.log2(sec.L/secLambda / 0.1)
            numSeg = math.ceil(2**numSeg_log2)
            if numSeg % 2 == 0:
                numSeg += 1
            sec.nseg = numSeg

            #debugging print statements
            #print("\nlength of section", sec, "is now", (sec.L/sec.nseg)/secLambda, "lambda")
            #print("fixed by using a total of", sec.nseg, "segments")
    return

def createElectrode(somaSection, pySectionList, neuron_name=None):
    electrodeSec = h.Section()
    electrodeSec.L = 10
    electrodeSec.diam = 1
    electrodeSec.connect(somaSection, 0)

    pySectionList.append(electrodeSec)

    # equivCylAxon.connect(axonEnd, 1)
    # print(equivCylAxon(0.5).area())

    return pySectionList, electrodeSec

def initializeModel(morph_file, neuron_name):

    cell = instantiate_swc(morph_file)

    allSections_nrn = h.SectionList()
    for sec in h.allsec():
        allSections_nrn.append(sec=sec)
    
    # Create a Python list from this SectionList
    # Select sections from the list by their index

    allSections_py = [sec for sec in allSections_nrn]    

    #extra sectionLists if necessary for visualization purposes
    colorR = h.SectionList()
    colorB = h.SectionList()
    colorK = h.SectionList()
    colorG = h.SectionList()
    colorV = h.SectionList()

    if neuron_name == "DNp01":
        sizIndex = 0
    elif neuron_name == "DNp02":
        sizIndex = 2
    elif neuron_name == "DNp03":
        sizIndex = 2
    elif neuron_name == "DNp04":
        sizIndex = 4
    elif neuron_name == "DNp06":
        sizIndex = 4
    elif neuron_name == "LPLC2": # HERE IS WHERE YOU CHANGE THE SIZ INDEX NUMBER/LOCATION OF THE SIZ IN THE MODEL.
        sizIndex = 0

    axonList = h.SectionList()
    tetherList = h.SectionList()
    dendList = h.SectionList()

    for sec in allSections_py:
        if "soma" in sec.name():
            somaSection = sec
            colorG.append(somaSection)
        elif "axon" in sec.name():
            axonList.append(sec)
        elif "dend_11" in sec.name():
            tetherList.append(sec)
        elif "dend_6" in sec.name():
            axonEnd = sec
        else:
            dendList.append(sec)
    i = 0
    for sec in axonList:
        #if i <= 4:
        if i == sizIndex:
            sizSection = sec
        i += 1
    colorV.append(sizSection)

    allSections_py, electrodeSec = createElectrode(somaSection, allSections_py, neuron_name)

    shape_window = h.PlotShape(h.SectionList(allSections_py))           # Create a shape plot
    shape_window.exec_menu('Show Diam')    # Show diameter of sections in shape plot
    shape_window.color_all(9)

    shape_window.color_list(axonList, 2)
    shape_window.color_list(colorG, 4)  
    shape_window.color_list(tetherList, 1)
    shape_window.color_list(dendList, 3)
    shape_window.color_list(colorV, 7)


    if neuron_name == "DNp01":
        erev = -66.6544#-72.5
        change_memCap(memcap=4.167)#3.5531)
        change_Ra(ra=17.6461)#30.4396)
        change_gLeak(gleak= 0.0011196588, erev=erev)#0.0012663288, erev=erev)
        # change_Ra(34)#33.2617)
        # change_gLeak(3.4e-4, erev=erev)#4.4e-9)    
        # change_memCap(1)#1.4261)
    elif neuron_name == "DNp02": 
        erev = -70.8#-70.812 TOOK FIRST 5000 DATAPOINTS FROM EACH HJ CURRENT STEP AND TOOK MEAN
        # change_Ra(91)
        # change_gLeak(0.0002415, erev=erev)  
        # change_memCap(1)
        change_memCap(memcap=1.595)
        change_Ra(ra=34.4495)
        change_gLeak(gleak=0.000147726, erev=erev)
    elif neuron_name == "DNp03":
        erev = -58.75#68
        # change_Ra(33)
        # change_gLeak(gleak=0.00034, erev=erev)
        # change_memCap(1)
        change_memCap(memcap=1.595)
        change_Ra(ra=34.4495)
        change_gLeak(gleak=0.000147726, erev=erev)
    elif neuron_name == "DNp04":
        erev=-72.5
        change_memCap(memcap=1.595)
        change_Ra(ra=34.4495)
        change_gLeak(gleak=0.000147726, erev=erev)
    elif neuron_name == "DNp06":
        erev = -60#-60.0406 SEE DNP02
        # change_Ra(91)
        # change_gLeak(0.0002415, erev=erev)  
        # change_memCap(1)
        change_memCap(memcap=1.595)
        change_Ra(ra=34.4495)
        change_gLeak(gleak=0.000147726, erev=erev)
    else:
        erev=-72.5
        change_Ra()
        change_gLeak()
        change_memCap()

    nsegDiscretization(allSections_py)
    

    return cell, allSections_py, allSections_nrn, somaSection, sizSection, erev, axonList, tetherList, dendList, shape_window, electrodeSec


#####################
# Simple simulations creating and inserting an electrode into the soma/SIZ and recording. 

def activateCurrentInjectionSIZ(erev, sizSection, somaSection, electrodeSec, continueRun=150, current=None, injDur=None, delay= None):

    h.v_init = erev   # Resting membrane potential 

    t_vec = h.Vector()
    t_vec.record(h._ref_t)

    stimobj = h.IClamp(sizSection(0.5))
    stimobj.delay = delay
    stimobj.dur = injDur
    stimobj.amp = current
    stimobj.i = 0

    SIZsec = h.Vector()
    SIZsec.record(sizSection(0.0551186)._ref_v)
    # vfakeSynSIZ = fakeSynSIZ.to_python()
    # vfakeSynSIZ = numpy.array(vfakeSynSIZ)

    Somasec = h.Vector()
    Somasec.record(somaSection(0.5)._ref_v)

    Electrodesec = h.Vector()
    Electrodesec.record(electrodeSec(0.5)._ref_v)

    h.finitialize(erev * mV)
    h.continuerun(continueRun * ms)


    plt.plot(t_vec, SIZsec, 'r')
    plt.plot(t_vec, Somasec, 'g')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axhline(y=erev+10, color='black', linestyle='--')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.plot(t_vec, Electrodesec, 'b')
    plt.legend(['SIZ', 'Soma', '0 mV', '10mV from resting', "ElectrodeSec"])
    maxSIZ_depol = SIZsec.max()
    print(erev)
    print(Somasec.max() - erev)
    maxSIZ_depol = SIZsec.max()
    print(f'Membrane potential at max SIZ depol: ' + str(maxSIZ_depol))
    print(f'Max SIZ depol: ' + str(maxSIZ_depol -erev))
    print(f'Membrane potential change at Soma: ' + str(Somasec.max() - erev))

    plt.title('Depolarization of DNp01 soma in response to SIZ current injection, SIZ depolarized to {} mV, SIZ depolarized to {} mV above resting'.format(round(maxSIZ_depol, 3), round(Somasec.max() - erev, 3)))
    # plt.savefig('DNp01_finalSIZfakeSpike.svgz', format='svgz')
    plt.show()

def activateCurrentInjectionSOMA(erev, sizSection, somaSection, electrodeSec, continueRun=150, current=None, injDur=None, delay= None):

    h.v_init = erev   # Resting membrane potential 

    t_vec = h.Vector()
    t_vec.record(h._ref_t)

    stimobj = h.IClamp(somaSection(0.5))
    stimobj.delay = delay
    stimobj.dur = injDur
    stimobj.amp = current
    stimobj.i = 0

    SIZsec = h.Vector()
    SIZsec.record(sizSection(0.0551186)._ref_v)
    # vfakeSynSIZ = fakeSynSIZ.to_python()
    # vfakeSynSIZ = numpy.array(vfakeSynSIZ)

    Somasec = h.Vector()
    Somasec.record(somaSection(0.5)._ref_v)

    Electrodesec = h.Vector()
    Electrodesec.record(electrodeSec(0.5)._ref_v)

    h.finitialize(erev * mV)
    h.continuerun(continueRun * ms)


    plt.plot(t_vec, SIZsec, 'r')
    plt.plot(t_vec, Somasec, 'g')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axhline(y=erev+10, color='black', linestyle='--')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')
    plt.plot(t_vec, Electrodesec, 'b')
    plt.legend(['SIZ', 'Soma', '0 mV', '10mV from resting', "ElectrodeSec"])
    maxSIZ_depol = SIZsec.max()
    print(f'Membrane potential at max SIZ depol: ' + str(maxSIZ_depol))
    print(f'Max SIZ depol: ' + str(maxSIZ_depol -erev))
    print(f'Membrane potential change at Soma: ' + str(Somasec.max() - erev))
    print(f'Ratio of Soma max amplitude/SIZ max amplitude: ' + str((Somasec.max() - erev)/(maxSIZ_depol -erev )) )
    

    plt.title('Depolarization of DNp01 SIZ in response to SOMA current injection, SIZ depolarized to {} mV, SIZ depolarized to {} mV above resting'.format(round(maxSIZ_depol, 3), round(SIZsec.max() - erev, 3)))
    plt.show()

#####################


def main():
    neuron_name = "LPLC2"

    # Tk().withdraw()
    # fd_title = "Select morphology file to initialize"
    # morph_file = fd.askopenfilename(filetypes=[("swc file", "*.swc"), ("hoc file","*.hoc")], initialdir=r"datafiles/morphologyData", title=fd_title) #This only works if you have a directory set up 

    # Here insert the path file for your .swc morphology, it will automatically load up and intialize your model.
    morph_file = ("C:/Users/antho/Documents/Downloads/720575940622093546_obaid.swc")

    cell, allSections_py, allSections_nrn, somaSection, sizSection, erev, axonList, tetherList, dendList, shape_window, electrodeSec = initializeModel(morph_file, neuron_name)
    
    ###FOR AMS FINAL PASSIVE PROPERTY ASSIGNMENT###

    # raVal = 17.6461#10.3294#15
    # gleakVal = 0.0011196588
    # cmVal = 4.167
    if neuron_name == "LPLC2": 
        erev = -66.4298 - 0.2
        raVal = 212#350#266.1
        gleakVal = 1/2300#1/1600#1800
        cmVal = 0.7#0.77#0.8

    elec_raVal = 235.6                  
    # elec_gleakVal = 0
    elec_cmVal = 6.4

    sealCon_8GOhm = 0.0003978
    # sealCon_2GOhm = 0.0016
    elec_gleakVal = sealCon_8GOhm

    #Here you can changes the values for the passive properties of the LPLC2, change the values above and the functions will handle there rest. 
    change_Ra(ra=raVal, electrodeSec=electrodeSec, electrodeVal = elec_raVal)
    change_gLeak(gleak=gleakVal, erev=erev, electrodeSec=electrodeSec, electrodeVal = elec_gleakVal)
    change_memCap(memcap=cmVal, electrodeSec=electrodeSec, electrodeVal = elec_cmVal)


    ### Running the simulations with injecting at the SOMA and SIZ.
    # I have arbitrarily set the SIZ so these functions work, but you will need to adjust that. 
    #Here you can change around the injection sites (soma vs SIZ), the current strength, duration, and the delay of when the simulation starts. 
    activateCurrentInjectionSOMA(erev, sizSection, somaSection, electrodeSec, continueRun=50, current=.001, injDur=5, delay= 5)
    #activateCurrentInjectionSIZ(erev, sizSection, somaSection, electrodeSec, continueRun=50, current=107.5, injDur=.1, delay= 5)

#Runs the actual simulation. 
main()