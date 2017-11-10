import os
from array import array
import SampleTypeLUT
import datetime
import keras
from math import sqrt
import numpy.lib.recfunctions as recfunctions
from common import *
import ROOT

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.Reset()


#ROOT.gROOT.SetBatch(1)
#ROOT.gStyle.SetStatW(0.07)
#ROOT.gStyle.SetStatH(0.06)

#ROOT.gStyle.SetOptStat(0)

#ROOT.gStyle.SetErrorX(0)
#ROOT.gStyle.SetErrorY(0)

#ROOT.gStyle.SetTitleStyle(0)
#ROOT.gStyle.SetTitleAlign(13) ## coord in top left
#ROOT.gStyle.SetTitleX(0.)
#ROOT.gStyle.SetTitleY(1.)
#ROOT.gStyle.SetTitleW(1)
#ROOT.gStyle.SetTitleH(0.058)
#ROOT.gStyle.SetTitleBorderSize(0)

#ROOT.gStyle.SetPadLeftMargin(0.126)
#ROOT.gStyle.SetPadRightMargin(0.10)
#ROOT.gStyle.SetPadTopMargin(0.06)
#ROOT.gStyle.SetPadBottomMargin(0.13)

#ROOT.gStyle.SetMarkerStyle(1)



def writeNNToTree(input_file, weight, modellist, modelLUT, sampletype, masslist, outFile):
    isSignal = (sampletype<13 or sampletype>=100)
    cut = "(91 - ll_M) > 15.0"
    FullInputs = ['jj_pt', 'll_pt', 'll_M', 'll_DR_l_l', 'jj_DR_j_j', 'llmetjj_DPhi_ll_jj', 'llmetjj_minDR_l_j', 'llmetjj_MTformula','mt2', 'jj_M','isSF','hme_h2mass_reco','isElEl','isElMu','isMuEl','isMuMu','cross_section','event_weight_sum']
    #dataset, weight = tree_to_numpy(f, FullInputs, weight_expression, cut, reweight_to_cross_section=False)
    file_handle = TFile.Open(input_file)

    tree = file_handle.Get('t')

    """
    ## not normalize to cross section
    cross_section = 1
    relative_weight = 1
    if not isSignal:
        print input_file
	relative_weight = getReweight_to_xsec(tree)
	print "reweight ",relative_weight

    if isinstance(weight, dict):
        # Keys are regular expression and values are actual weights. Find the key matching
        # the input filename
        found = False
        weight_expr = None
        if '__base__' in weight:
            weight_expr = weight['__base__']

        for k, v in weight.items():
            if k == '__base__':
                continue

            groups = re.search(k, input_file)
            if not groups:
                continue
            else:
                if found:
                    raise Exception("The input file is matched by more than one weight regular expression. %r" % input_file)

                found = True
                weight_expr = join_expression(weight_expr, v)

        if not weight_expr:
            raise Exception("Not weight expression found for input file %r" % weight_expr)

        weight = weight_expr

    """
    # Read the tree and convert it to a numpy structured array
    a = tree2array(tree, branches= FullInputs + [weight], selection=cut)
    # Rename the last column to 'weight'
    #print "print a type ", type(a)," shape ",a.shape, " len(a) ",len(a), " a.dtype.names ",a.dtype.names, " a[:10,:] ", a[:10]
    a.dtype.names = FullInputs + ['final_total_weight']

  
    #print "print a type ", type(a)," shape ",a.shape, " a.dtype.names ",a.dtype.names, " a[:10,:] ", a[:10]
    
    
    ## add mass column
    """
    resonant_mass_probabilities = [[260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 800, 900], [ 0.0384008 ,  0.03985746,  0.04913783,  0.0611964 ,  0.07281586, 0.08038202,  0.09056594,  0.09963943,  0.10548293,  0.11182466, 0.12333857,  0.12735809]]

    masscol = np.array([0.0]*len(a))
    if isSignal:
	mass = SampleTypeLUT.getSignalMass(sampletype)
    	masscol  = np.array([mass]*len(a))
    else:
	#rs = np.random.RandomState(42)
	#indices = rs.choice(len(resonant_mass_probabilities[0]), len(a), p=resonant_mass_probabilities[1])
	#masscol = np.array(np.take(resonant_mass_probabilities[0], indices, axis=0), dtype='float')
	
	masscol = np.array([mass]*len(a))
    """
    allnnout = [] 
    nnoutname = []
    
    #sample_type = ROOT.vector('string')()
    for i, key in enumerate(modellist):
	modelfile = os.path.join(modelLUT[key]['workingdir'], "hh_resonant_trained_model.h5")
	inputs = modelLUT[key]['inputs'] 
	thismodel = keras.models.load_model(modelfile)
        for mass in masslist:	
	    thisdataset = np.array(a[inputs].tolist(), dtype=np.float32)
	    masscol = [mass]*len(a)
	    thisdataset = np.c_[thisdataset, masscol]
	    #print "this model: ",key," inputs ",inputs, " datasetshape ",thisdataset.shape," type ",type(thisdataset)
	    nnout = thismodel.predict(thisdataset, batch_size=5000, verbose=1)
	    nnout = np.delete(nnout, 1, axis=1).flatten()
	    allnnout.append(nnout)
	    nnoutname.append("nnout_%s_M%d"%(key,mass))
    
    #for i in range(len(modellist)):
    for i in range(len(allnnout)):
	    #recfunctions.append_fields(base, names, data=None, dtypes=None, fill_value=-1, usemask=True, asrecarray=False)
	    a = recfunctions.append_fields(a, nnoutname[i], allnnout[i])
	    #a = recfunctions.(a, nnoutname[i], allnnout[i])

    #dataset.dtype.names = FullInputs + ['weight', 'mass'] + nnoutname
    #print "print a type ", type(a)," shape ",a.shape, " a.dtype.names ",a.dtype.names, " a[:10,:] ", a[:10]
    print "\n a dytpenames ", a.dtype.names 

    tfile = ROOT.TFile(outFile, "RECREATE")
    #ch_new = array2tree(dataset,  FullInputs + ['mass'] + nnoutname + ['finalweight'], "evtreeHME_nn")
    ch_new = array2tree(a, "evtreeHME_nn")
    
    ch_new.GetCurrentFile().Write() 
    ch_new.GetCurrentFile().Close()

def bestSensitivityNNWP(ch_signal, ch_bg, todraw, nbins, cuts_signal, cuts_bg, mass, outputdir):
    histbg = ROOT.TH1F("histbg","histbg",nbins, 0.0, 1.0)
    ch_bg.Draw(todraw+">> histbg",cuts_bg)
    histsignal = ROOT.TH1F("histsignal","histsignal",nbins, 0.0, 1.0)
    ch_signal.Draw(todraw+">> histsignal",cuts_signal)
    
    bestWP = 0.0
    bestS = 0.0
    for i in range(1, nbins+1):
	nbg = histbg.Integral(i, nbins)
	nsignal = histsignal.Integral(i, nbins)
	if (nbg==0):
		continue
	if nsignal/sqrt(nbg+nsignal) > bestS:
		bestS = nsignal/sqrt(nbg+nsignal)
		bestWP = histbg.GetXaxis().GetBinCenter(i)
    if bestS == 0.0:
	    print "error in calculating S/sqrt(B) ",bestS," nncut ",bestWP
    else:
	    print "nnout ",todraw, " nncut ",bestWP," Sensitivity ",bestS 
    #makeplot = True
    if bestS>0.0:
	ctmp = ROOT.TCanvas("NNWP","NNWP",800, 600)
	#print "histbg ",histbg.Integral()," histsignal ",histsignal.Integral()
	ratio = histsignal.Integral()/histbg.Integral()
	histbg.Scale(ratio)
	histbg.Rebin(5)
	histbg.SetLineColor(600)
	#histsignal.Scale(1.0/histsignal.Integral())
	histsignal.Rebin(5)
	histsignal.SetLineColor(632)
	histbg.SetStats(0)
	histbg.Draw()
	histsignal.Draw("same")
	histbg.SetTitle("Signal and Background DNN output distribution, Mass=%d GeV"%mass)
	histbg.GetYaxis().SetTitle("arbitary unit")
	histbg.GetXaxis().SetTitle("DNN output")
	
        legend = ROOT.TLegend(0.55,0.65,0.88,0.85)
	legend.SetHeader("DNN training: kinematics+"+todraw[6:])
	ens = legend.AddEntry(histsignal, "Signal","pl")
	ens.SetTextColor(632)
	enbg = legend.AddEntry(histbg, "Background","pl")
	enbg.SetTextColor(600)
	legend.Draw("same")
	#c1.SaveAs(outputdir+"bestSensitivityNNWP_%s_nnoutcut%d_bestS%d.png"%(todraw, bestWP*100, bestS*1000))
	ctmp.SaveAs(outputdir+"bestSensitivityNNWP__mass%d_%s_nnoutcut%d.png"%(mass, todraw, bestWP*100))
    return bestWP, bestS

def hist1D(ch_signal, ch_bg, todraw, bins, xtitle, cuts_signallist, cuts_bglist, colors, title, legs, isNormalized, plotname):

    #color = [ROOT.kRed-4, ROOT.kBlue-4, ROOT.kMagenta-2, ROOT.kOrange-3, ROOT.kSpring, ROOT.kCyan, ROOT.kGreen+2]
    #color = [ROOT.kRed, ROOT.kBlue, ROOT.kMagenta+2, ROOT.kGreen+2, ROOT.kCyan+2, ROOT.kOrange+2, ROOT.kSpring]
    
    histbglist = []; histsignallist = []
    for i in range(len(cuts_signallist)):
	
        histbg = ROOT.TH1F("histbg%d"%i,"histbg",bins[0], bins[1], bins[2])
        ch_bg.Draw(todraw+">> "+histbg.GetName(),cuts_bglist[i])
        histsignal = ROOT.TH1F("histsignal%d"%i,"histsignal", bins[0], bins[1], bins[2])
        ch_signal.Draw(todraw+">> "+histsignal.GetName(),cuts_signallist[i])
	if histbg.GetEntries() == 0 or histsignal.GetEntries() == 0:
		print "error in ",plotname," todraw ",todraw," histbg entry ",histbg.GetEntries()," cut ",cuts_bglist[i], " hist_s entry ",histsignal.GetEntries(), " cut ",cuts_signallist[i] 
        if isNormalized:
	    histbg.Scale(1.0/histbg.Integral())
	    histsignal.Scale(1.0/histsignal.Integral())
	histbglist.append(histbg)
	histsignallist.append(histsignal)
    c1 = ROOT.TCanvas()
    c1.SetGridx(); c1.SetGridy(); c1.SetTickx(); c1.SetTicky()
    hs = ROOT.THStack("hs", title)

    deltay = 0.04*(len(cuts_signallist)+1)

    legend = ROOT.TLegend(0.45,0.65,0.88,0.66+deltay);
    legend.SetHeader("DNN training: kinematics+")
    #legend.SetFillColor(ROOT.kWhite); 
    #legend.SetTextSize(0.05); legend.SetTextFont(62)
    for i in range(len(cuts_signallist)):
	hs.Add(histbglist[i]) 
	hs.Add(histsignallist[i])
	histbglist[i].SetLineColor(colors[i])
	histbglist[i].SetLineStyle(1)
	histbglist[i].SetLineWidth(2)
	histsignallist[i].SetLineColor(colors[i])
	histsignallist[i].SetLineStyle(4)
	histsignallist[i].SetLineWidth(2)
	entry = legend.AddEntry(histsignallist[i], legs[i],"")
	entry.SetTextColor(colors[i])
	#legend.AddEntry(histbglist[i], legs[i]+",Background","pl")
    tex = ROOT.TLatex(0.6,0.2, "signal: - - - background: ---")
    tex.SetNDC()
    tex.SetTextSize(.035)
    hs.Draw("nostackhist")
    hs.GetHistogram().GetXaxis().SetTitle(xtitle)
    if isNormalized:
	    hs.GetHistogram().GetYaxis().SetTitle("Normalized to unity")

    legend.Draw("same")
    tex.Draw("same")
    c1.SaveAs(plotname+".C")
    c1.SaveAs(plotname+".pdf")

def hist2D(ch, todraw, xbins, ybins, cuts, title, xtitle, ytitle, tex, plotname):
    c1 = ROOT.TCanvas()
    c1.Clear()
    hist = ROOT.TH2F("hist","hist",xbins[0], xbins[1], xbins[2], ybins[0], ybins[1], ybins[2])
    ch.Draw(todraw+">> hist", cuts)
    #print "todraw ",todraw, " cuts ",cuts," Integral ",hist.Integral()
    #warning: by using data driven method in DY estimation, some bin content after weighting is negative
    hist.SetMinimum(0.0)
    hist.Draw("colz")
    #hist.Draw()
    tex.Draw("same")
    hist.SetTitle(title)
    hist.GetXaxis().SetTitle(xtitle)
    hist.GetYaxis().SetTitle(ytitle)
    c1.SaveAs(plotname+".C")
    c1.SaveAs(plotname+".pdf")

def plotSB(file_slist, file_blist, nnlist, colors, title, mass, outdir, plotsuffix):
    ch_s = ROOT.TChain("evtreeHME_nn")
    ch_b = ROOT.TChain("evtreeHME_nn")
	
    for file_s in file_slist:
	ch_s.Add(file_s)
    for file_b in file_blist:
        ch_b.Add(file_b)
    print "file_s ",file_slist," file_b ",file_blist
    #bgweight = 708040.0/441719.0 ###  some jobs failed in processing ttbar 
    bgweight = 1.0#no job failed in this ntuples production
    variables = {"mt2":{
	    		"latexname":"M_{T2}",
			"bins":[80, 0.0, 500.0]
			},
		 "hme_h2mass_reco":{
			 "latexname":"HME Reco Mass",
			 "bins":[80, 200.0, 1000.0]
			 },
		 }
    legs = []
    nncutlist = []
    cuts_signallist = []
    cuts_backgroundlist = []
    #cuts_signal = "(35.87*1000*cross_section/event_weight_sum)*final_total_weight*(isMuMu && hme_h2mass_reco>250)"
    #cuts_bg = "(35.87*1000*cross_section/event_weight_sum)*final_total_weight*(isMuMu && hme_h2mass_reco>250)*%f"%(bgweight)
    signalweight = "final_total_weight"
    bgweight = "(35.87*1000*cross_section/event_weight_sum)*final_total_weight*%f"%(bgweight)
    for nnout in nnlist:
	cut = "isMuMu"# channel cut
	if "HME" in nnout:
            cut = cut+" && hme_h2mass_reco>250.0 "
	cuts_signal = "(%s)*(%s)"%(signalweight, cut)
	cuts_bg = "(%s)*(%s)"%(bgweight, cut)
	#cuts_signal = "(35.87*1000*cross_section/event_weight_sum)*final_total_weight*(isMuMu && hme_h2mass_reco>250)"
	#cuts_bg = "(35.87*1000*cross_section/event_weight_sum)*final_total_weight*(isMuMu && hme_h2mass_reco>250)*%f"%(bgweight)
	
	nncut, nnSensitivity = bestSensitivityNNWP(ch_s, ch_b, nnout, 200, cuts_signal, cuts_bg, mass, outdir)
        legs.append(nnout[6:]+", DNN cut: %.3f"%(nncut))
	print "Singal ",plotsuffix," nncut ",nncut, " sensitivity ",nnSensitivity
	nncutlist.append(nncut)
	cuts_signallist.append("(%s)*(%s && %s>%f)"%(signalweight, cut, nnout, nncut))
	cuts_backgroundlist.append("(%s)*(%s && %s>%f)"%(bgweight, cut, nnout, nncut))
	#cuts_signallist.append("(35.87*1000*cross_section/event_weight_sum)*final_total_weight*(isMuMu && hme_h2mass_reco>250 && %s>%f)"%(nnout, nncut))
	#cuts_backgroundlist.append("(35.87*1000*cross_section/event_weight_sum)*final_total_weight*%f*(isMuMu && hme_h2mass_reco>250 && %s>%f)"%(bgweight, nnout, nncut))
    for var in variables:
	hist1D(ch_s, ch_b, var, variables[var]["bins"],  variables[var]["latexname"], cuts_signallist, cuts_backgroundlist, colors, title+","+variables[var]["latexname"], legs, True, os.path.join(outdir, var+"_afterDNN_"+plotsuffix))
	for i, nnout in enumerate(nnlist):
	    hist1D(ch_s, ch_b, var, variables[var]["bins"],  variables[var]["latexname"], [cuts_signallist[i]], [cuts_backgroundlist[i]], [colors[i]], title+","+variables[var]["latexname"], [legs[i]], True, os.path.join(outdir, var+"_afterDNN_"+plotsuffix+"_"+nnout))
    #### 2D hist
    for i, nnout in enumerate(nnlist):
	tex = ROOT.TLatex(0.7,0.2, nnout[6:])
        tex.SetNDC()
        tex.SetTextSize(.04)
        tex.SetTextFont(62)
	tex.SetTextColor(colors[i])
	hist2D(ch_s, "hme_h2mass_reco:mt2", variables["mt2"]["bins"], variables["hme_h2mass_reco"]["bins"], cuts_signallist[i]+"*(hme_h2mass_reco>0)", title+", plotting Signal", variables["mt2"]["latexname"], variables["hme_h2mass_reco"]["latexname"], tex,  os.path.join(outdir,"mt2VsHME_afterDNN_"+plotsuffix+"_"+nnout+"_signal"))
	hist2D(ch_b, "hme_h2mass_reco:mt2", variables["mt2"]["bins"], variables["hme_h2mass_reco"]["bins"], cuts_backgroundlist[i]+"*(hme_h2mass_reco>0)", title+", plotting Background", variables["mt2"]["latexname"], variables["hme_h2mass_reco"]["latexname"], tex, os.path.join(outdir,"mt2VsHME_afterDNN_"+plotsuffix+"_"+nnout+"_background"))
    for var in variables:
        for i, nnout in enumerate(nnlist):
	    tex = ROOT.TLatex(0.6,0.13, nnout[6:])
            tex.SetNDC()
            tex.SetTextSize(.04)
            tex.SetTextFont(62)
	    tex.SetTextColor(colors[i])

	    hist2D(ch_s, var+":"+nnout, [53, -0.02,1.04], variables[var]["bins"], signalweight+"*(%s>0.0)"%(var), title+", plotting Signal", "DNN output", variables[var]["latexname"], tex,  os.path.join(outdir, "nnoutVs"+var+"_beforeDNNcut_"+plotsuffix+"_"+nnout+"_signal"))
	    hist2D(ch_b, var+":"+nnout, [53, -0.02,1.04], variables[var]["bins"], bgweight+"*(%s>0.0)"%(var), title+", plotting Background", "DNN output", variables[var]["latexname"], tex,  os.path.join(outdir, "nnoutVs"+var+"_beforeDNNcut_"+plotsuffix+"_"+nnout+"_background"))


#extraweight is used for compensating the "HLT" safe ID cut
channelcuts = {"MuMu":{"cut":"isMuMu","extraweight": 24827.9/23891.14, "latex":"#mu#mu"},
		"MuEl":{"cut":"(isMuEl || isElMu)", "extraweight": 25786.8/30211.9, "latex":"e#mu"},
		"ElEl":{"cut":"isElEl","extraweight":8487.6/12190.1, "latex":"ee"},
		}
###
def histForlimits1D(backgrounddict, bgnames, filesignal, mass, todraw, cut, xbins,  outfile, plotname):
    chlist = {}
    colors = [628, 596, 820, 432, 418]
    for key in backgrounddict:
	ch = ROOT.TChain("evtreeHME_nn")
	for f in backgrounddict[key]:
	    ch.Add(f)
	chlist[key] = ch
    ch_s =  ROOT.TChain("evtreeHME_nn"); ch_s.Add(filesignal)
    rfile = ROOT.TFile(outfile, "UPDATE")
    weight = "(35.87*1000*cross_section/event_weight_sum)*final_total_weight"
    weight_s = "(35.87*1000*5.0/event_weight_sum)*final_total_weight"
    for channel in channelcuts:
	allhists = []
        hs = ROOT.THStack("allbg_M%d"%mass+channel, "  ")
        legend = ROOT.TLegend(0.7,0.65,0.88,0.65+len(backgrounddict)*.06); 
	legend.SetTextSize(0.045); legend.SetTextFont(62)
        #legend.SetHeader("DNN training: kinematics+")
	BGSum = 0.0

	hist_data_fake = ROOT.TH1F("data_obs_"+channel+"_M%d"%mass, "data_obs_"+channel+"_M%d"%mass, xbins[0], xbins[1], xbins[2])
	for i, key in enumerate(bgnames):
	    hist = ROOT.TH1F(key+"_"+channel+"_M%d"%mass, key+"_"+channel+"_M%d"%mass, xbins[0], xbins[1], xbins[2])
	    finalcut = "("+ cut + " && "+ channelcuts[channel]["cut"] +")*"+weight+"*%f"%(channelcuts[channel]["extraweight"])
	    chlist[key].Draw(todraw + ">> " + hist.GetName(), finalcut)
	    hs.Add(hist)
	    hist.SetFillColor(colors[i])
	    allhists.append(hist)
	    print "mass ",mass, " channel ", channel," bg ",key," rate ",hist.Integral()
	    BGSum = BGSum + hist.Integral()
	    hist_data_fake.Add(hist)
	#for hist,name in zip(reversed(allhists), reversed(bgnames)):
	for i in range(len(allhists)):
	    legend.AddEntry(allhists[len(allhists)-i-1], bgnames[len(allhists)-i-1], "f")
	    allhists[len(allhists)-i-1].Write()
	hs.Write()
	
        hist_s = ROOT.TH1F("signal_"+channel+"_M%d"%mass, "signal_"+channel+"_M%d"%mass, xbins[0], xbins[1], xbins[2])
	finalcut_s  = "("+ cut + " && "+ channelcuts[channel]["cut"] +")*"+weight_s+"*%f"%(channelcuts[channel]["extraweight"])
	ch_s.Draw(todraw + ">> " + hist_s.GetName(), finalcut_s)
	hist_data_fake.Add(hist_s)
	print "mass ",mass, " channel ", channel," rate:signal ",hist_s.Integral()," BG(TT,DY,sT) ",BGSum," data(fake) ",hist_data_fake.Integral()
	hist_s.SetLineColor(1)
	hist_s.SetLineWidth(2)
	hist_s.Write()
	hist_data_fake.Write()
        ens = legend.AddEntry(hist_s, "Signal, M=%d"%mass, "l")
	ens.SetTextSize(.03)

        c1 = ROOT.TCanvas()
        c1.Clear()
	hs.Draw("hist")
	hist_s.Draw("samehist")
	
	tex1 = ROOT.TLatex(0.17,0.8, channelcuts[channel]["latex"]+" channel")
	tex1.SetNDC(); tex1.SetTextSize(.045)
	tex2 = ROOT.TLatex(0.19,0.5, "M_{jj}<75 GeV "+"  "*5+" 75 GeV <=M_{jj}<140 GeV"+ "  "*5+" M_{jj} >= 140 GeV")
	tex2.SetNDC(); tex2.SetTextSize(.033)
	tex1.Draw("same")
	tex2.Draw("same")
	#hs.GetHistogram().SetTitle(" #scale[1.4]{#font[61]{CMS}} Simulation Preliminary"+"  "*38+"35.87 fb^{-1} (13 TeV),2016")
	hs.GetHistogram().SetTitle("")
	#hs.GetHistogram().SetTitleSize(.04)
	#hs.GetHistogram().SetTitleOffset(1.2)
	tex0 = ROOT.TLatex(0.1,0.91, " #scale[1.4]{#font[61]{CMS}} Simulation Preliminary"+"  "*14+"35.87 fb^{-1} (13 TeV),2016")
	tex0.SetNDC(); tex0.SetTextSize(.04); tex0.SetTextFont(42)
	tex0.Draw("same")
        hs.GetHistogram().GetXaxis().SetTitle("DNN output, M_{jj} bins")
	hs.GetHistogram().GetYaxis().SetTitle("Events")

        c1.Update()

        legend.Draw("same")
	#tex.Draw("same")
        c1.SaveAs(plotname+"_RadionM%d_"%mass+channel+".C")
        c1.SaveAs(plotname+"_RadionM%d_"%mass+channel+".pdf")
	
	     


    
    

#filepath = "/Users/taohuang/Documents/DiHiggs/20170530/20171021_Louvain/"
#filepath = "/Users/taohuang/Documents/DiHiggs/20170530/testfile/"
filepath = "/Users/taohuang/Documents/DiHiggs/20170530/20171102_Louvain/"
#outdir = "/Users/taohuang/Documents/DiHiggs/20170530/20171021_Louvain_addNN/"
outdir = filepath
suffix = 'addNN'
output_suffix = '{:%Y-%m-%d}_{}'.format(datetime.date.today(), suffix)
#output_suffix = '2017-11-08_addNN'
output_folder = os.path.join(outdir, output_suffix)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

#modellist = ['MTonly']
modellist = ['MTonly','MT2only','MTandMT2','MTandMT2_MJJ','MTandMT2_HME','MTandMT2_HMEMJJ', 'MTandMJJ']
modellist = ['MTonly','MTandMJJ']
allfiles = os.listdir(filepath)
#allfiles = ['TT_all.root']
def makeNtuple_prediction(masses):
     
    for f in allfiles:
	if f.endswith(".root"):
		name = f.split('.')[0]
		outname = ''
		try:
			sampletype =  SampleTypeLUT.sampletypedict[name]
		except KeyError:
			##background
			sampletype =  SampleTypeLUT.sampletypedict[name[:-4]]
			#outname = name+"_addNN_M%s.root"%mass
		#addsampleType(filepath+f, sampletype, name+"_addType.root")
		#weight = "event_weight * trigeff * jjbtag_heavy * jjbtag_light * llidiso * pu"
		weight = "total_weight"
		if name.startswith("DY"):
			weight = weight+"* dy_nobtag_to_btagM_weight"
		print 'file ',file,' name ',name,' type ',type(sampletype)," sampletype ",sampletype," weight ",weight
		if sampletype>=100:
		    outname = name+"_addNN.root"
		    output_file = os.path.join(output_folder, outname)		
		    mass = SampleTypeLUT.getSignalMass(sampletype)
		    writeNNToTree(filepath+f, weight, modellist, SampleTypeLUT.ModelLUT, sampletype, [mass], output_file)
		else:
			#for mass in masses:
		    #outname = name+"_addNN_M%s.root"%mass
		    outname = name+"_addNN.root"
		    output_file = os.path.join(output_folder, outname)		
		    writeNNToTree(filepath+f, weight, modellist, SampleTypeLUT.ModelLUT, sampletype, masses, output_file)



#makeNtuple_prediction([260])
makeNtuple_prediction([260, 300, 350, 400, 450, 500, 600,750, 800, 900])
#makeNtuple_prediction([260, 350, 400, 500])
#makeNtuple_prediction(350)
#makeNtuple_prediction(400)
#makeNtuple_prediction(450)
#makeNtuple_prediction(600)
#makeNtuple_prediction(500)

nnlist = ["nnout_MTonly", "nnout_MT2only","nnout_MTandMT2","nnout_MTandMT2_MJJ","nnout_MTandMT2_HME","nnout_MTandMT2_HMEMJJ","nnout_MTandMJJ"]
nnlist = ["nnout_MTonly", "nnout_MTandMJJ"]
#nnlist = ["nnout_MTonly"]
plotdir = "plotsafterDNN/"
def makeplots(nnlist, plotdir, masspoints):
    os.system("mkdir -p "+plotdir)
    #masspoints = [260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 800, 900]
    #colors = [632, 600, 618, 418, 434, 802]
    colors = [628, 596, 614, 797, 820, 432, 418]
    for mass in masspoints: 
	file_slist = []
        file_slist.append(output_folder+"/radion_M%d_addNN.root"%(mass))
	bgnamelist = ["TT","DYM10to50","DYToLL0J","DYToLL1J","DYToLL2J","sT_top","sT_antitop"]
	#bgnamelist = ["DYM10to50","DYToLL0J","DYToLL1J","DYToLL2J"]
	file_blist = []
	for bgname in bgnamelist:
		file_blist.append( output_folder+"/%s_all_addNN_M%d.root"%(bgname ,mass) )
	title = "Signal, Radion M=%d GeV "%mass
        plotSB(file_slist, file_blist, nnlist, colors, title, mass, plotdir, "radion_M%d"%mass)

#makeplots(nnlist, "plotsafterDNN_splitted_20171102/", [400])
#makeplots(nnlist, "plotsafterDNN_splitted_20171102/", [260, 350, 400, 500])

def makeLouvainLimithist(masspoints, outdir):
    bgnames = ["sT","DY","TT"]
    outfile = os.path.join(outdir, "Hhh_FinalBGYield_5pbsignalindata.root")
    plotname = os.path.join(outdir, "Hhh_FinalBGYield")
    ###create tfile
    tfile = ROOT.TFile(outfile, "RECREATE")
    tfile.Close()
    todraw = "({nnout}*(jj_M<75)+(jj_M>=75 && jj_M<140)*({nnout}+1)+(jj_M>=140)*({nnout}+2))".format(nnout = "nnout_MTonly")
    cut = "1"
    nnbins = [75, 0.0, 3.0] 
    for mass in masspoints:
	file_s = os.path.join(output_folder, "radion_M%d_addNN.root"%(mass))
        backgrounddict = {}
	for bg in bgnames:
	    allfiles = os.listdir(output_folder)
	    backgrounddict[bg] = []
	    for f in allfiles:
		if f.startswith(bg) and "addNN_M%d"%mass in f:
		   backgrounddict[bg].append(os.path.join(output_folder, f))
        histForlimits1D(backgrounddict, bgnames, file_s, mass, todraw, cut, nnbins, outfile, plotname)

#limitdir = "HistForLimits_addsignalindata"
#os.system("mkdir -p "+limitdir)
#makeLouvainLimithist([260, 350, 400, 500], limitdir)
