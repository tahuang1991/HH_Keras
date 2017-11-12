import os
import ROOT
import re
import numpy as np
import datetime

def generatedatacard(outfile, rootfile, channels, processes, mass, addobservation):
    datacardfile = open(outfile, "write")
    imax = len(channels)
    jmax = len(processes)
    datacardfile.write("imax %d\n"%imax)
    datacardfile.write("jmax %d\n"%(jmax-1))
    datacardfile.write("kmax *\n")  ## uncertainty sources
    datacardfile.write("#-----------------------------------------------------------------------------\n")## comment out
    rootfilename = rootfile.split('/')[-1]
    datacardfile.write("shapes * * %s $PROCESS_$CHANNEL_M%d $PROCESS_$SYSTEMATIC\n"%(rootfilename, mass))
    datacardfile.write("#-----------------------------------------------------------------------------\n")## comment out
    ## channels
    datacardfile.write("bin\t\t")
    for chan in channels:
	datacardfile.write(chan+"\t")
    datacardfile.write("\n")
    rfile = ROOT.TFile(rootfile, "READ") ## read rate from histogram
    if addobservation:
        datacardfile.write("observation\t\t")
	for chan in channels:
	    hist = rfile.Get("data_obs_{channel}_M{mass}".format(channel = chan, mass = mass))
	    datacardfile.write("{rate}\t".format(rate = hist.Integral()))
	datacardfile.write("\n")
    else:
        datacardfile.write("#observation, not available yet\n")
    datacardfile.write("#-----------------------------------------------------------------------------\n")## comment out
    ##start to set rate
    nratelines = 4
    ratelineheads = ["bin","process","process","rate"]
    for iline,head in enumerate(ratelineheads):
        datacardfile.write("{head}\t\t".format(head = head))
        for i, chan in enumerate(channels):
	    for j, process in enumerate(processes): 
                if iline == 0:
                    datacardfile.write("{channel}\t".format(channel = chan))
                elif iline == 1:
                    datacardfile.write("{process}\t".format(process = process))
                elif iline == 2:
                    datacardfile.write("%d\t"%(j))
                elif iline == 3:
                    hist = rfile.Get("{process}_{channel}_M{mass}".format(process = process, channel = chan, mass = mass))
                    datacardfile.write("%.1f\t"%(hist.Integral()))
        datacardfile.write("\n")
    datacardfile.write("#-----------------------------------------------------------------------------\n")## comment out
    ## add uncertainty, should be done in a smarter way?
    funcertainty = open("allUncertainties.txt","read")
    for line in funcertainty:
        datacardfile.write(line)
     
    rfile.Close()
    datacardfile.close()
    #os.system("cat "+outfile)

def alldatacards(rootfile, masspoints, addobservation, outdir):
    os.system("cp %s %s"%(rootfile, outdir))
    channels = ["MuMu","MuEl","ElEl"]
    processes = ["signal","TT","DY","sT"]
    for mass in masspoints:
        outfile = os.path.join(outdir, "shape_M%d.txt"%mass)
        generatedatacard(outfile, rootfile, channels, processes, mass, addobservation)

def submitdatacards(cardname, method):
    logfile = "expectedlimits_tmp.log"
    os.system("combine {cardname}  -M {method} -t -1 > {logfile}".format(cardname = cardname, method = method, logfile = logfile))
    logopen = open(logfile, "read")
    sigma_xsec = 5000.0
    percents = [2.5, 16.0, 50.0, 84.0, 97.5]
    limits = {}
    for line in logopen:
        if not line.startswith("Expected"):
            continue
        #print "line ",line
        for percent in percents:
            pattern = "Expected%5.1f%%: r < "%(percent)
            #print " percent ",percent, " pattern ",pattern
            if re.match(pattern, line):
                limits[percent] = float(line.split(" ")[-1].rstrip()) * sigma_xsec
    #print "limits ",limits
    return limits

def Limitplots(cardnameprefix, masspoints, method, plotname):
    onesigma_up = []
    twosigma_up = []
    onesigma_low = []
    twosigma_low = []
    central = []
    method = "Asymptotic"
    for mass in masspoints:
        cardname = cardnameprefix + "_M%d.txt"%mass
        limits = submitdatacards(cardname, method)
        print "cardname ",cardname," limits ",limits,"\n"
        central.append(limits[50.0])
        twosigma_low.append(limits[2.5])
        onesigma_low.append(limits[16.0])
        onesigma_up.append(limits[84.0])
        twosigma_up.append(limits[97.5])

    fakeerrors = [0.0]*len(masspoints)
    #g_onesigma = TGraphAsymmErrors(len(masspoints),  np.array(masspoints),  np.array(central), np.array(fakeerrors), np.array(fakeerrors), np.array(onesigma_low), np.array(onesigma_up))
    #g_twosigma = TGraphAsymmErrors(len(masspoints),  np.array(masspoints),  np.array(central), np.array(fakeerrors), np.array(fakeerrors), np.array(twosigma_low), np.array(twosigma_up))
    c1 = ROOT.TCanvas("c1","c1",800, 600)
    c1.SetLogy()
    c1.SetGridx()  
    c1.SetGridy()  
    c1.SetTickx()  
    c1.SetTicky()  
    onesigma_low.reverse()
    twosigma_low.reverse()
    onesigma_all = onesigma_up + onesigma_low
    twosigma_all = twosigma_up + twosigma_low
    masspoints_all = masspoints + list(reversed(masspoints))
    masspoints_f =  np.array(masspoints)+0.0
    print "allmasspoints ",masspoints_all," onesigma ",onesigma_all," float masspoints ",masspoints_f
    g_central = ROOT.TGraph(len(masspoints),  np.array(masspoints)+0.0,  np.array(central))
    g_onesigma = ROOT.TGraph(len(masspoints)*2,  np.array(masspoints_all)+0.0,  np.array(onesigma_all))
    g_twosigma = ROOT.TGraph(len(masspoints)*2,  np.array(masspoints_all)+0.0,  np.array(twosigma_all))


    g_twosigma.SetFillColor(ROOT.kYellow)
    g_twosigma.SetLineStyle(2)
    g_onesigma.SetFillColor(ROOT.kSpring)
    g_onesigma.SetLineStyle(2)
    g_central.SetLineWidth(2)
    g_central.SetLineStyle(7)

    b1 = ROOT.TH1F("b2","b2",14, 250.0, 950.0)
    #b1.SetTitle(" #scale[1.4]{#font[61]{CMS}} Simulation Preliminary"+"  "*14+"35.87 fb^{-1} (13 TeV),2016")
    b1.SetTitle(" ")
    b1.GetYaxis().SetTitle("95% C.L. limits on production rate (fb)")
    b1.GetXaxis().SetTitle("M_{X} [GeV]")
    yhigh = max(twosigma_up)*1.2
    ylow = min(twosigma_low)*.9
    b1.GetYaxis().SetRangeUser(ylow, yhigh)
    b1.SetStats(0)

    #g_twosigma.SetLineColor(ROOT.kBlack)
    #g_onesigma.SetLineColor(ROOT.kBlack)
    b1.Draw()
    g_twosigma.Draw("fe3same")
    g_onesigma.Draw("fe3same")
    g_central.Draw("same")
    
    leg = ROOT.TLegend(0.6,0.68,0.9,0.90)
    leg.SetFillColor(ROOT.kWhite)
    leg.SetTextFont(42)
    #leg.AddEntry(grdata,"observed","pl")
    leg.AddEntry(g_central,"Expected 95% upper limit","l")
    leg.AddEntry(g_onesigma,"1 std. deviation","f")
    leg.AddEntry(g_twosigma,"2 std. deviation","f")
    tex0 = ROOT.TLatex(0.1,0.91, " #scale[1.4]{#font[61]{CMS}} Simulation Preliminary"+"  "*14+"35.87 fb^{-1} (13 TeV),2016")
    tex0.SetNDC(); tex0.SetTextSize(.04); tex0.SetTextFont(42)
    tex0.Draw("same")
    
    leg.Draw("same")
    c1.SaveAs(plotname+"_95Upperlmit_Nodata_tmp.pdf")
    c1.SaveAs(plotname+"_95Upperlmit_Nodata_tmp.C")
    #c1.SaveAs(plotname+"_95Upperlmit_Nodata_tmp.png")



suffix = 'Radion'
output_suffix = '{:%Y-%m-%d}_{}'.format(datetime.date.today(), suffix)
datacarddir = "Datacards/"
#os.system("mkdir -p "+datacarddir)
datacarddir = "./"
outdir = os.path.join(datacarddir, output_suffix)
os.system("mkdir -p "+outdir)
method = "Asymptotic"
#rootfile = "HistForLimits/Hhh_FinalBGYield_5pbsignalindata.root"
rootfile = "Hhh_FinalBGYield.root"
masslist = [260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 750, 800, 900]
#alldatacards(rootfile, masslist, False, outdir)
cardnameprefix = os.path.join(outdir, "shape")
plotname = "Radion"
masslist = [260, 270, 300, 350, 400, 450, 500, 550, 600, 650, 750]
Limitplots(cardnameprefix, masslist, method, plotname)
