import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cbook as cbook
from sklearn import metrics

# ROC
Path1       = 'hh_resonant_trained_models_NewNtuples_HME/2017-08-10_90epochs/'
Path1b      = Path1+"splitted_by_mass/"
List_csv_1  = [[Path1+'roc_curve_X.cvs',Path1+'roc_curve_Y.cvs'],[Path1b+'roc_curve_fixed_M_260_X.cvs',Path1b+'roc_curve_fixed_M_260_Y.cvs'],[Path1b+'roc_curve_fixed_M_300_X.cvs',Path1b+'roc_curve_fixed_M_300_Y.cvs'],[Path1b+'roc_curve_fixed_M_350_X.cvs',Path1b+'roc_curve_fixed_M_350_Y.cvs'], [Path1b+'roc_curve_fixed_M_400_X.cvs',Path1b+'roc_curve_fixed_M_400_Y.cvs'],[Path1b+'roc_curve_fixed_M_450_X.cvs',Path1b+'roc_curve_fixed_M_450_Y.cvs'],[Path1b+'roc_curve_fixed_M_600_X.cvs',Path1b+'roc_curve_fixed_M_600_Y.cvs'],[Path1b+'roc_curve_fixed_M_800_X.cvs',Path1b+'roc_curve_fixed_M_800_Y.cvs'],[Path1b+'roc_curve_fixed_M_900_X.cvs',Path1b+'roc_curve_fixed_M_900_Y.cvs']]
Path2       = 'hh_resonant_trained_models_NewNtuples/2017-08-10_90epochs/'
Path2b      = Path2+"splitted_by_mass/"
List_csv_2  = [[Path2+'roc_curve_X.cvs',Path2+'roc_curve_Y.cvs'],[Path2b+'roc_curve_fixed_M_260_X.cvs',Path2b+'roc_curve_fixed_M_260_Y.cvs'],[Path2b+'roc_curve_fixed_M_300_X.cvs',Path2b+'roc_curve_fixed_M_300_Y.cvs'],[Path2b+'roc_curve_fixed_M_350_X.cvs',Path2b+'roc_curve_fixed_M_350_Y.cvs'], [Path2b+'roc_curve_fixed_M_400_X.cvs',Path2b+'roc_curve_fixed_M_400_Y.cvs'],[Path2b+'roc_curve_fixed_M_450_X.cvs',Path2b+'roc_curve_fixed_M_450_Y.cvs'],[Path2b+'roc_curve_fixed_M_600_X.cvs',Path2b+'roc_curve_fixed_M_600_Y.cvs'],[Path2b+'roc_curve_fixed_M_800_X.cvs',Path2b+'roc_curve_fixed_M_800_Y.cvs'],[Path2b+'roc_curve_fixed_M_900_X.cvs',Path2b+'roc_curve_fixed_M_900_Y.cvs']]
Path3       = 'hh_resonant_trained_models_NewNtuples_HME_Nentr/2017-08-15_70epochs/'
Path3b      = Path3+"splitted_by_mass/"
List_csv_3  = [[Path3+'roc_curve_X.cvs',Path3+'roc_curve_Y.cvs'],[Path3b+'roc_curve_fixed_M_260_X.cvs',Path3b+'roc_curve_fixed_M_260_Y.cvs'],[Path3b+'roc_curve_fixed_M_300_X.cvs',Path3b+'roc_curve_fixed_M_300_Y.cvs'],[Path3b+'roc_curve_fixed_M_350_X.cvs',Path3b+'roc_curve_fixed_M_350_Y.cvs'], [Path3b+'roc_curve_fixed_M_400_X.cvs',Path3b+'roc_curve_fixed_M_400_Y.cvs'],[Path3b+'roc_curve_fixed_M_450_X.cvs',Path3b+'roc_curve_fixed_M_450_Y.cvs'],[Path3b+'roc_curve_fixed_M_600_X.cvs',Path3b+'roc_curve_fixed_M_600_Y.cvs'],[Path3b+'roc_curve_fixed_M_800_X.cvs',Path3b+'roc_curve_fixed_M_800_Y.cvs'],[Path3b+'roc_curve_fixed_M_900_X.cvs',Path3b+'roc_curve_fixed_M_900_Y.cvs']]
output_name = ['Roc_global.pdf','Roc_260_global.pdf','Roc_300_global.pdf','Roc_350_global.pdf','Roc_400_global.pdf','Roc_450_global.pdf','Roc_600_global.pdf','Roc_800_global.pdf','Roc_900_global.pdf']
output_dir  = 'Comparison'

for iR in range(len(List_csv_1)):
  x1 = np.genfromtxt(List_csv_1[iR][0], delimiter=',')
  y1 = np.genfromtxt(List_csv_1[iR][1], delimiter=',')
  x2 = np.genfromtxt(List_csv_2[iR][0], delimiter=',')
  y2 = np.genfromtxt(List_csv_2[iR][1], delimiter=',')
  x3 = np.genfromtxt(List_csv_3[iR][0], delimiter=',')
  y3 = np.genfromtxt(List_csv_3[iR][1], delimiter=',')
  fig = plt.figure(1, figsize=(7, 7), dpi=300)
  fig.clear()

  ax = fig.add_subplot(111) # Create an axes instance
  ax.plot(x1, y1, '-', color='blue', lw=2, label="with HME")
  ax.plot(x2, y2, '-', color='green', lw=2, label="no HME")
  ax.plot(x3, y3, '-', color='red', lw=2, label="HME, HME #entr.")
  ax.margins(0.05)
  #if iR==0:
  ax.set_xlim([-0.01,0.4]);
  ax.set_ylim([0.4,1.01]);
  ax.set_xlabel("Background efficiency")
  ax.set_ylabel("Signal efficiency")
  fig.set_tight_layout(True)
  ax.legend(loc='center right', numpoints=1, frameon=False)
  fig.savefig(os.path.join(output_dir, output_name[iR]))
  plt.close()
