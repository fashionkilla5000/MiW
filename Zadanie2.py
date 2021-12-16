import collections
import easygui
import pandas as pd
import os
import numpy as np
from numpy import nan

def wczytajPlik():
    sciezka = easygui.fileopenbox()
    if "normalized" in sciezka:
        return pd.read_csv(sciezka, header=None, na_values='?', delim_whitespace=True)
    else:
        with open(sciezka) as f:
            lines = f.readlines()
        return lines


def main():

    #data = wczytajPlik()
    config = wczytajPlik()
    #print(data)


    nazwa_datasetu = config[0].strip("\n").split(": ")[1]
    klasa_decyzyjna = config[1].strip("\n").split(": ")[1]
    rows = config[2].strip("\n").split(": ")[1]
    cols = config[3].strip("\n").split(": ")[1]
    symboliczne = ""
    try:
        symboliczne = config[4].strip("\n").split(": ")[1]
    except:
        pass
    przedzial_normalizacji = config[5].strip("\n").split(": ")[1]
    minim = config[5].strip("\n").split(": ")[1].strip("[").strip("]").split(",")[0]
    maxim = config[5].strip("\n").split(": ")[1].strip("[").strip("]").split(",")[1]

    if symboliczne != "":
        print("Dane symboliczne znajdują się w kolumnach: "+symboliczne)
    print("Przedział normalizacji to: " + minim, maxim)
    float(input("Podaj atrybuty: "))

main()