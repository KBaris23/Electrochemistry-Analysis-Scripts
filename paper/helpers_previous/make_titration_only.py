import os,re,shutil,sys,pandas as pd
sp=r"C:\Users\Asus\AppData\Local\Temp\claude\C--TEMP-BO\1190eb47-9db3-44d1-8d53-499f90d1ac2c\scratchpad"
SRC={"kana_try":r"500um_planar_BO_try_again_20260918_112055\planar_BO_kana_20260918_112056","kana_st2":r"500um_planar_kana_20260917_170330\500um_planar_kana_20260917_170349","amp0":r"100um_amp0_high_conc_20260915_130213\amp0_100um_20260915_130234","vanco":r"vanco_first_try_20260826_132944\vanco_try_1_20260826_132945"}
for n,rel in SRC.items():
    src=os.path.join(r"C:\TEMP\BO",rel); dst=os.path.join(r"C:\TEMP\BO\titration_only",n); os.makedirs(dst+r"\methods_used",exist_ok=True)
    hashes=set(pd.read_csv(f"{sp}\\tit2_{n}.csv").hash)
    c=0
    for f in os.listdir(src):
        m=re.match(r"swv_ch\d+_([0-9a-f]+)_meas_.*\.csv$",f)
        if m and m.group(1) in hashes:
            shutil.copy2(os.path.join(src,f),dst); c+=1
            ms=os.path.join(src,"methods_used",f[:-4]+".ms")
            if os.path.exists(ms): shutil.copy2(ms,dst+r"\methods_used")
    print(n,"copied csv",c,"ms",len(os.listdir(dst+r"\methods_used")))
