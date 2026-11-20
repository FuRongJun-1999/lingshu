#!/usr/bin/env python3
"""Reproduce pendulum release accuracy, RK4, compatibility and timing evidence.

Requires Python 3.12+ and mpmath (validated with 1.4.1). From the repository root,
after installing mpmath in a separate environment, run:

    python bench/repro/pendulum_release_accuracy.py --output ./pendulum-evidence

Use --repo PATH when invoked elsewhere, --baseline REF to change the comparison
commit (default d74848f), and optionally --expected-fixed-sha HEX to require an
exact LF-normalized SHA-256 for lingshu/world/physics.py. The current working
source and baseline are frozen under the requested output directory. Production
sources are only read; no package installation or Git checkout is performed.

Independent oracle: 80-digit sn(K-u, m), with m=k*k and k=sin(theta0/2). The
pendulum convention is DLMF 22.19.2 (K+u is equivalent by time symmetry):
https://dlmf.nist.gov/22.19.E2 . AGM/Landen recurrences are DLMF 22.20.1 and
22.20.4: https://dlmf.nist.gov/22.20.E1 and https://dlmf.nist.gov/22.20.E4 .
The stable descent identity a_n**2-c_n**2=b_n**2 is derived directly from
22.20.1; atan2(c*sin(phi), hypot(b,c*cos(phi))) is an algebraic reformulation
of the principal asin in 22.20.4, not an additional formula quoted from DLMF.

JSON retains every input, oracle value, error, phase decomposition, RK4 sample,
timing sample, environment and source hash. report.md is a readable summary.
Timing is descriptive on a shared host, not a hardware-independent guarantee.
"""
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,json,math,os,platform,random,statistics,subprocess,sys,time
from pathlib import Path
import mpmath as mp
ROOT=None
REPO=None
FIXED_SOURCE=None
BASE="d74848f"
EXPECTED=None
DPS=80

def write_json(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
def sha(data):return hashlib.sha256(data).hexdigest()
def lfsha(data):return sha(data.replace(b"\r\n",b"\n"))

def load_source(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def capture_sources(expected):
 baseline=subprocess.run(["git","-C",str(REPO),"show",BASE+":lingshu/world/physics.py"],check=True,stdout=subprocess.PIPE).stdout
 fixed=FIXED_SOURCE.read_bytes()
 if expected is not None and lfsha(fixed)!=expected:raise RuntimeError("Fixed production source drifted: expected "+expected+", got "+lfsha(fixed))
 if expected is None:expected=lfsha(fixed)
 directory=ROOT/"sources";directory.mkdir(exist_ok=True);bp=directory/("physics_baseline_"+BASE[:12]+".py");fp=directory/("physics_fixed_"+expected[:12]+".py");bp.write_bytes(baseline);fp.write_bytes(fixed)
 meta={"baseline":{"commit":BASE,"snapshot":str(bp),"sha256":sha(baseline),"lf_sha256":lfsha(baseline)},"fixed":{"working_source":str(FIXED_SOURCE),"snapshot":str(fp),"sha256":sha(fixed),"lf_sha256":lfsha(fixed),"expected_lf_sha256":expected},"runner_lf_sha256":lfsha(Path(__file__).read_bytes())}
 meta["repository"]=str(REPO);meta["git_head"]=subprocess.run(["git","-C",str(REPO),"rev-parse","HEAD"],check=True,stdout=subprocess.PIPE,text=True).stdout.strip()
 meta["fixed_source_git_status"]=subprocess.run(["git","-C",str(REPO),"status","--porcelain","--","lingshu/world/physics.py"],check=True,stdout=subprocess.PIPE,text=True).stdout.strip()
 write_json(ROOT/"source_manifest.json",meta);return load_source("pendulum_baseline_frozen",bp),load_source("pendulum_fixed_frozen",fp),meta

def period(theta):
 theta=mp.mpf(theta);k=mp.sin(theta/2);return 4*mp.ellipk(k*k)
def reference(theta,omega,t):
 # Convert each incoming binary64 value exactly before multiplying. Do not feed
 # the production rounded modulus or production rounded omega*t to the oracle.
 theta,w,tt=mp.mpf(theta),mp.mpf(omega),mp.mpf(t);k=mp.sin(theta/2);m=k*k;quarter=mp.ellipk(m);u=w*tt
 if theta==0 or u==0:return theta,quarter,u,k
 value=2*mp.asin(k*mp.ellipfun("sn",quarter-u,m))
 if isinstance(value,mp.mpc):
  if abs(value.imag)>mp.mpf("1e-60"):raise RuntimeError("Unexpected non-real oracle")
  value=value.real
 return value,quarter,u,k

def invoke(module,theta,omega,t,explicit=True):
 try:
  v=module.pendulum_exact_angle(theta,omega,t,tol_deg=180.) if explicit else module.pendulum_exact_angle(theta,omega,t)
  return {"value":v,"finite":math.isfinite(v)}
 except Exception as e:return {"error_type":type(e).__name__,"error":str(e)}

def phase_metrics(module,theta,omega,t,quarter,u,k,expected,actual):
 complementary=math.cos(.5*theta);qf=math.pi/(2*module.agm(1.,complementary));uf=omega*t;reduced=math.remainder(uf,4*qf);p=4*quarter
 difference=mp.mpf(reduced)-u;wrapped=difference-p*mp.floor(difference/p+mp.mpf("0.5"));phase_error=abs(wrapped)
 value_at_reduced=2*mp.asin(k*mp.ellipfun("sn",quarter-mp.mpf(reduced),k*k));value_at_reduced=mp.re(value_at_reduced)
 intrinsic=abs(mp.mpf(actual)-value_at_reduced);bound=2*abs(k)*phase_error+mp.mpf("5e-12")
 return {"implemented_quarter_u":qf,"reference_quarter_u":mp.nstr(quarter,85),"quarter_error_u":float(abs(mp.mpf(qf)-quarter)),"product_rounding_error_u":float(abs(mp.mpf(uf)-u)),"implemented_remainder_u":reduced,"circular_phase_error_u":float(phase_error),"intrinsic_error_at_implemented_phase_rad":float(intrinsic),"phase_lipschitz_plus_5e_12_bound_rad":float(bound),"within_phase_bound":abs(mp.mpf(actual)-expected)<=bound}

def accuracy(base,fixed):
 rows=[];violations=[];amplitudes=[("zero",0.),("tiny",1e-12),("0.001deg",math.radians(.001)),("1deg",math.radians(1)),("5deg",math.radians(5)),("15deg",math.radians(15)),("30deg",math.radians(30)),("60deg",math.radians(60)),("90deg",math.radians(90)),("120deg",math.radians(120)),("170deg",math.radians(170)),("179deg",math.radians(179)),("179.999deg",math.radians(179.999)),("179.99999999deg",math.radians(179.99999999)),("pi_minus_1e-12",math.pi-1e-12),("nextafter_pi",math.nextafter(math.pi,0.))]
 factors=[0.,.125,.25,.5,.75,1.,1.125,3.375,10.1,100.3]
 for label,positive in amplitudes:
  for sign in ((1.,) if positive==0 else (1.,-1.)):
   theta=sign*positive;full=period(theta)
   for omega in (.25,-1.,3.13):
    for factor in factors:
     # Construct a representable time from high-precision target periods. The
     # oracle is subsequently evaluated at that actual binary64 time, not target.
     t=float(mp.mpf(factor)*full/mp.mpf(abs(omega)))
     expected,quarter,u,k=reference(theta,omega,t);b=invoke(base,theta,omega,t);f=invoke(fixed,theta,omega,t)
     row={"amplitude":label,"theta0_rad":theta,"omega0":omega,"t":t,"target_period_fraction":factor,"reference_rad":mp.nstr(expected,85),"baseline":b,"fixed":f}
     for variant,result in (("baseline",b),("fixed",f)):
      if "value" in result:result["abs_error_rad"]=float(abs(mp.mpf(result["value"])-expected))
     if "value" not in f:violations.append({"reason":"fixed_exception","case":len(rows),"row":row})
     else:
      v=f["value"];f["phase"]=phase_metrics(fixed,theta,omega,t,quarter,u,k,expected,v)
      checks={"finite":f["finite"],"amplitude_bound":abs(v)<=abs(theta)+1e-13,"time_even":invoke(fixed,theta,omega,-t).get("value")==v,"omega_even":invoke(fixed,theta,-omega,t).get("value")==v,"theta_odd":abs(invoke(fixed,-theta,omega,t).get("value",math.inf)+v)<1e-13,"phase_error_bound":f["phase"]["within_phase_bound"],"short_horizon_abs_error":factor>3.375 or f["abs_error_rad"]<=1e-11}
      row["checks"]=checks
      for name,ok in checks.items():
       if not ok:violations.append({"reason":name,"case":len(rows),"amplitude":label,"theta":theta,"omega":omega,"t":t,"factor":factor,"error_rad":f["abs_error_rad"],"phase":f["phase"]})
     rows.append(row)
  print("accuracy "+label,flush=True)
 # Long horizons characterize float input/product/period-reduction limits.
 long=[]
 for label,theta in [(x,y) for x,y in amplitudes if x in ("5deg","120deg","179.999deg","nextafter_pi")]:
  full=period(theta)
  for cycles in (1e3,1e6,1e9,1e12,1e15):
   for omega in (1.,-3.13):
    t=float((mp.mpf(cycles)+mp.mpf("0.127"))*full/mp.mpf(abs(omega)));expected,quarter,u,k=reference(theta,omega,t);f=invoke(fixed,theta,omega,t);b=invoke(base,theta,omega,t)
    row={"amplitude":label,"theta0_rad":theta,"omega0":omega,"t":t,"target_cycles":cycles,"reference_rad":mp.nstr(expected,85),"baseline":b,"fixed":f}
    for result in (f,b):
     if "value" in result:result["abs_error_rad"]=float(abs(mp.mpf(result["value"])-expected))
    if "value" in f:
     f["phase"]=phase_metrics(fixed,theta,omega,t,quarter,u,k,expected,f["value"])
     if not f["phase"]["within_phase_bound"]:violations.append({"reason":"long_horizon_phase_bound","row":row})
    else:violations.append({"reason":"long_horizon_exception","row":row})
    long.append(row)
 summary={"short_cases":len(rows),"long_cases":len(long),"fixed_checks":sum(len(r.get("checks",{})) for r in rows)+len(long),"fixed_violations":len(violations),"violations":violations,"short_max_abs_error_rad":max(r["fixed"].get("abs_error_rad",0.) for r in rows),"short_3_375_period_max_abs_error_rad":max(r["fixed"].get("abs_error_rad",0.) for r in rows if r["target_period_fraction"]<=3.375),"baseline_numeric_cases":sum("value" in r["baseline"] for r in rows),"baseline_exceptions":sum("error_type" in r["baseline"] for r in rows),"baseline_max_abs_error_rad":max(r["baseline"].get("abs_error_rad",0.) for r in rows)}
 write_json(ROOT/"accuracy_raw.json",{"precision_decimal_digits":DPS,"summary":summary,"grid":rows,"long_horizons":long});return summary,rows,long

def rk4(theta,u,max_step):
 steps=max(1,math.ceil(abs(u)/max_step));h=u/steps;position=theta;velocity=0.
 for _ in range(steps):
  a1=velocity;b1=-math.sin(position)
  a2=velocity+.5*h*b1;b2=-math.sin(position+.5*h*a1)
  a3=velocity+.5*h*b2;b3=-math.sin(position+.5*h*a2)
  a4=velocity+h*b3;b4=-math.sin(position+h*a3)
  position+=(h/6.)*(a1+2*a2+2*a3+a4);velocity+=(h/6.)*(b1+2*b2+2*b3+b4)
 return position,velocity,steps

def rk4_checks(fixed):
 rows=[]
 for degrees in (1.,5.,30.,90.,120.,170.,179.):
  theta=math.radians(degrees);full=period(theta)
  for sign in (1.,-1.):
   for fraction in (.125,.25,.625,1.125,3.375):
    u=float(sign*mp.mpf(fraction)*full);expected,_,_,_=reference(theta,1.,u);coarse=rk4(theta,u,1/512);fine=rk4(theta,u,1/4096);f=invoke(fixed,theta,1.,u)
    energy=.5*fine[1]**2+(1-math.cos(fine[0]));initial=1-math.cos(theta)
    row={"degrees":degrees,"u":u,"period_fraction":fraction,"coarse_step_max":1/512,"fine_step_max":1/4096,"coarse_steps":coarse[2],"fine_steps":fine[2],"coarse_abs_error_rad":float(abs(mp.mpf(coarse[0])-expected)),"fine_abs_error_rad":float(abs(mp.mpf(fine[0])-expected)),"fine_energy_error":abs(energy-initial),"fixed_abs_error_rad":float(abs(mp.mpf(f["value"])-expected)),"fixed_vs_fine_rad":abs(f["value"]-fine[0]),"fine_within_1e_8":abs(mp.mpf(fine[0])-expected)<mp.mpf("1e-8")}
    rows.append(row)
 # The nextafter(pi,0) double RK4 case demonstrates why a binary64 numerical
 # integrator cannot serve as the near-separatrix oracle: sub-ulp increments.
 theta=math.nextafter(math.pi,0.);u=float(period(theta)/4);r=rk4(theta,u,1/4096);expected,_,_,_=reference(theta,1.,u)
 extreme={"theta0_rad":theta,"u":u,"rk4_theta":r[0],"reference_rad":mp.nstr(expected,85),"rk4_error_rad":float(abs(mp.mpf(r[0])-expected)),"note":"Characterization only: binary64 RK4 can stall near the turning point; the 80-digit elliptic oracle remains the accuracy reference."}
 result={"cases":len(rows),"fine_max_abs_error_rad":max(x["fine_abs_error_rad"] for x in rows),"fixed_max_abs_error_rad":max(x["fixed_abs_error_rad"] for x in rows),"fine_over_1e_8":sum(not r["fine_within_1e_8"] for r in rows),"rows":rows,"near_separatrix_characterization":extreme}
 write_json(ROOT/"rk4_raw.json",result);return result

def timing(label,fn,calls=256,repeats=21,warmups=5):
 wall=[];cpu=[]
 for i in range(warmups+repeats):
  gc.collect();on=gc.isenabled();gc.disable()
  try:
   t0,c0=time.perf_counter_ns(),time.process_time_ns()
   for _ in range(calls):fn()
   dt,dc=time.perf_counter_ns()-t0,time.process_time_ns()-c0
  finally:
   if on:gc.enable()
  if i>=warmups:wall.append(dt/calls/1e3);cpu.append(dc/calls/1e3)
 values=sorted(wall);median=statistics.median(wall)
 return {"label":label,"unit":"microseconds per call","calls_per_sample":calls,"warmups":warmups,"repeats":repeats,"wall_us_samples":wall,"cpu_us_samples":cpu,"median_us":median,"p95_us":values[math.ceil(.95*len(values))-1],"min_us":min(wall),"max_us":max(wall),"spread_max_minus_min_us":max(wall)-min(wall),"stdev_us":statistics.stdev(wall)}

def performance(base,fixed):
 rows=[]
 for degrees,explicit in ((1.,False),(5.,False),(30.,True),(90.,True),(120.,True),(170.,True),(179.999,True)):
  theta=math.radians(degrees);times=[float((mp.mpf(i)+mp.mpf("0.37"))*period(theta)/16) for i in range(16)];pair={"degrees":degrees,"explicit_tol_180":explicit}
  for name,module in (("baseline",base),("fixed",fixed)):
   def call(module=module):
    return [module.pendulum_exact_angle(theta,1.,t,tol_deg=180.) if explicit else module.pendulum_exact_angle(theta,1.,t) for t in times]
   probe=invoke(module,theta,1.,times[0],explicit)
   if "error" in probe:pair[name]=probe;continue
   result=timing("%s_%gdeg_%s"%(name,degrees,"tol180" if explicit else "default"),call,calls=128);result["unit"]="microseconds per 16-angle batch";result["median_us_per_angle"]=result["median_us"]/16;pair[name]=result
  if "median_us" in pair.get("baseline",{}) and "median_us" in pair.get("fixed",{}):pair["fixed_over_baseline"]=pair["fixed"]["median_us"]/pair["baseline"]["median_us"]
  rows.append(pair)
 # Fixed-only coverage where baseline rounds sin(theta/2) to one and rejects.
 theta=math.nextafter(math.pi,0.);u=float(period(theta)*mp.mpf("0.127"));rows.append({"degrees":"nextafter_pi","fixed":timing("fixed_nextafter_pi",lambda:fixed.pendulum_exact_angle(theta,1.,u),calls=1024),"baseline":invoke(base,theta,1.,u)})
 rk=[]
 for degrees in (5.,120.,170.):
  theta=math.radians(degrees);u=float(period(theta)/4);rk.append(timing("rk4_%gdeg_quarter_period_step_1_4096"%degrees,lambda:rk4(theta,u,1/4096),calls=1,repeats=11,warmups=3))
 write_json(ROOT/"performance_raw.json",{"workloads":rows,"rk4":rk,"limitations":["Wall timings are descriptive on a shared host; no CPU pinning/power control.","No high-precision oracle calls, imports or source capture occur inside timed blocks.","The 16-angle batch includes Python list construction and public API validation.","RK4 costs correspond to the stated numerical step, not equal-cost/equal-error alternatives."]});return rows,rk

def domains(fixed):
 checks=[]
 for name,args in [("theta_pi",(math.pi,1.,1.)),("theta_minus_pi",(-math.pi,1.,1.)),("theta_nan",(math.nan,1.,1.)),("omega_inf",(.1,math.inf,1.)),("time_nan",(.1,1.,math.nan)),("product_overflow",(.1,1e308,1e308))]:
  result=invoke(fixed,*args);checks.append({"name":name,"passed":"error" in result,"result":result})
 for omega in (0.,-0.):
  result=invoke(fixed,math.nextafter(math.pi,0.),omega,-100.);checks.append({"name":"zero_omega_%r"%omega,"passed":result.get("value")==math.nextafter(math.pi,0.),"result":result})
 return checks

def period_truth_checks(base,fixed):
 """Compare public truth periods against 80-digit K at declared binary64 theta.

  The API first converts its binary64 degree input with math.radians. Record
  that conversion's separate effect near the separatrix; do not mistake it
  for an error of the subsequent AGM period computation.
 """
 amplitudes=[1.,5.,120.,170.,179.999,179.99999999,math.degrees(math.pi-1e-12),math.degrees(math.nextafter(math.pi,0.))];configs=[]
 for degrees in amplitudes:
  for sign in (1.,-1.):
   for w in (.25,1.,3.13):configs.append({"theta0_deg":sign*degrees,"omega0":w})
 for w in (.25,1.,3.13):configs.append({"theta0_deg":0.,"omega0":w})
 for degrees in amplitudes[-3:]:
  for sign in (1.,-1.):
   for length in (.125,1.,10.):configs.append({"theta0_deg":sign*degrees,"length_m":length,"g_m_s2":9.81})
 rows=[]
 for config in configs:
  theta=math.radians(float(config["theta0_deg"]));w=math.sqrt(config["g_m_s2"]/config["length_m"]) if "length_m" in config else config["omega0"]
  k=mp.sin(mp.mpf(theta)/2);expected=4*mp.ellipk(k*k)/mp.mpf(w);exact_degree_theta=mp.mpf(config["theta0_deg"])*mp.pi/180;degree_k=mp.sin(exact_degree_theta/2);degree_reference=4*mp.ellipk(degree_k*degree_k)/mp.mpf(w)
  row={"declaration":config,"declared_theta0_rad_binary64":theta,"reference_period_s_85_digits":mp.nstr(expected,85),"degree_to_radian_conversion_abs_error_rad":float(abs(mp.mpf(theta)-exact_degree_theta)),"period_relative_difference_from_exact_degree_conversion":float(abs(expected-degree_reference)/degree_reference)}
  for name,module in (("baseline",base),("fixed",fixed)):
   try:
    value=module.pendulum_truth(config);actual=value.get("period_exact_elliptic",value["period_small_angle"]);error=abs(mp.mpf(actual)-expected);row[name]={"period_s":actual,"abs_error_s":float(error),"relative_error":float(error/expected),"theta0_rad":value["theta0_rad"],"omega0":value["omega0"]}
   except Exception as e:row[name]={"error_type":type(e).__name__,"error":str(e)}
  result=row["fixed"];row["passed"]="period_s" in result and result["relative_error"]<=1e-14 and result["theta0_rad"]==theta and result["omega0"]==w;rows.append(row)
 out={"cases":len(rows),"failed_cases":sum(not row["passed"] for row in rows),"fixed_max_relative_error":max(row["fixed"].get("relative_error",0.) for row in rows),"fixed_max_abs_error_s":max(row["fixed"].get("abs_error_s",0.) for row in rows),"baseline_exceptions":sum("error" in row["baseline"] for row in rows),"reference_convention":"80-digit elliptic K evaluated at the public declaration's math.radians binary64 theta; the distinct exact-real degree conversion effect is recorded, not silently attributed to AGM.","rows":rows}
 write_json(ROOT/"period_truth_raw.json",out);return out

def public_cn_compatibility(base,fixed):
 """This fix promises to retain the existing public cn numerical behavior."""
 rng=random.Random(20261120);cases=[]
 for k in (0.,.01,-.01,.5,-.5,.95,-.95,.999999,-.999999,math.nextafter(1.,0.),-math.nextafter(1.,0.)):
  for u in (0.,.001,-.001,.125,-.125,1.,-1.,2.,-2.,10.,-10.,100.,-100.,1000.,-1000.):cases.append((u,k))
 cases.extend((rng.uniform(-1000,1000),rng.uniform(-.999,.999)) for _ in range(2000));mismatches=[]
 for i,(u,k) in enumerate(cases):
  old,new=base.jacobi_cn(u,k),fixed.jacobi_cn(u,k)
  if old.hex()!=new.hex():mismatches.append({"index":i,"u":u,"k":k,"baseline_hex":old.hex(),"fixed_hex":new.hex()})
 out={"seed":20261120,"cases":len(cases),"bitwise_mismatches":len(mismatches),"mismatches":mismatches}
 write_json(ROOT/"public_cn_compatibility.json",out);return out

def main():
 global ROOT,REPO,FIXED_SOURCE,BASE
 p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
 p.add_argument("--repo",type=Path,default=Path(__file__).resolve().parents[2],help="Repository root (default: this script's repository root)")
 p.add_argument("--output",type=Path,required=True,help="Directory for JSON, frozen source copies and report.md")
 p.add_argument("--baseline",default=BASE,help="Git baseline ref (default: d74848f)")
 p.add_argument("--expected-fixed-sha",default=EXPECTED,help="Optional required LF-normalized fixed-source SHA-256")
 args=p.parse_args();REPO=args.repo.resolve();ROOT=args.output.resolve();FIXED_SOURCE=REPO/"lingshu/world/physics.py"
 if not FIXED_SOURCE.is_file():p.error("--repo does not contain lingshu/world/physics.py")
 BASE=subprocess.run(["git","-C",str(REPO),"rev-parse",args.baseline],check=True,stdout=subprocess.PIPE,text=True).stdout.strip()
 mp.mp.dps=DPS;ROOT.mkdir(parents=True,exist_ok=True)
 base,fixed,source=capture_sources(args.expected_fixed_sha);environment={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"processor":platform.processor(),"cpu_count":os.cpu_count(),"mpmath":mp.__version__,"mpmath_dps":mp.mp.dps,"pythonhashseed":os.environ.get("PYTHONHASHSEED"),"wall_clock_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"perf_counter":vars(time.get_clock_info("perf_counter")),"process_time":vars(time.get_clock_info("process_time"))};write_json(ROOT/"environment.json",environment)
 summary,grid,long=accuracy(base,fixed);rk=rk4_checks(fixed);perf,rkperf=performance(base,fixed);dc=domains(fixed);compat=public_cn_compatibility(base,fixed);periods=period_truth_checks(base,fixed);combined={"period_truth_summary":{k:v for k,v in periods.items() if k!="rows"},"public_jacobi_cn_compatibility":compat,"environment":environment,"sources":source,"accuracy_summary":summary,"domain_checks":dc,"rk4_summary":{k:v for k,v in rk.items() if k not in ("rows",)},"performance":perf,"rk4_performance":rkperf};write_json(ROOT/"comparison.json",combined)
 lines=["# Pendulum independent validation","","Fixed LF-normalized source SHA-256: "+source["fixed"]["lf_sha256"],"","Baseline: "+BASE,"","Reference: mpmath %s at %d decimal digits, theta=2 asin(k sn(K-u,m)), m=k², k=sin(theta0/2), K=ellipk(m). Each binary64 input is promoted exactly before the high-precision product. This does not reuse production AGM/Landen code."%(mp.__version__,DPS),"","Accuracy summary:",""]+["- %s: %s"%(k,v) for k,v in summary.items() if k!="violations"]
 lines += ["","| Amplitude | Through 100.3 periods: max error rad | Through 3.375 periods: max error rad |","|---|---:|---:|"]
 for label in dict.fromkeys(r["amplitude"] for r in grid):
  selected=[r for r in grid if r["amplitude"]==label];lines.append("| %s | %.6g | %.6g |"%(label,max(r["fixed"].get("abs_error_rad",0.) for r in selected),max(r["fixed"].get("abs_error_rad",0.) for r in selected if r["target_period_fraction"]<=3.375)))
 lines += ["","Long horizons: input product rounding and approximate-period reduction limit phase accuracy. The analytical speed bound |dtheta/du|<=2|k| yields |angle error|<=2|k| times circular phase error plus the sampled evaluation allowance 5e-12 rad. This is a checked sample decomposition, not a global floating-point proof.","","| Target cycles | Max angle error rad | Max circular phase error u | Max intrinsic error at implemented phase rad |","|---:|---:|---:|---:|"]
 for cycles in (1e3,1e6,1e9,1e12,1e15):
  selected=[r for r in long if r["target_cycles"]==cycles];lines.append("| %.0e | %.6g | %.6g | %.6g |"%(cycles,max(r["fixed"].get("abs_error_rad",0.) for r in selected),max(r["fixed"].get("phase",{}).get("circular_phase_error_u",0.) for r in selected),max(r["fixed"].get("phase",{}).get("intrinsic_error_at_implemented_phase_rad",0.) for r in selected)))
 lines += ["","RK4: %d signed-time cases, fine max error %.6g rad; fixed max error %.6g rad; fine errors >1e-8: %d. Energy and coarse/fine samples are retained in rk4_raw.json."%(rk["cases"],rk["fine_max_abs_error_rad"],rk["fixed_max_abs_error_rad"],rk["fine_over_1e_8"]),"","Binary64 RK4 near nextafter(pi,0): "+json.dumps(rk["near_separatrix_characterization"],ensure_ascii=False),"","Timing (microseconds per 16-angle batch; 5 warmups, 21 retained samples, 128 batches/sample):","","| Degrees / domain | Baseline median | Fixed median | Fixed p95 | Fixed / baseline |","|---|---:|---:|---:|---:|"]
 for row in perf:
  if "fixed_over_baseline" in row:lines.append("| %s / %s | %.3f | %.3f | %.3f | %.3fx |"%(row["degrees"],"tol180" if row["explicit_tol_180"] else "default",row["baseline"]["median_us"],row["fixed"]["median_us"],row["fixed"]["p95_us"],row["fixed_over_baseline"]))
 lines += ["","Limitations:","","- Shared host, uncontrolled CPU affinity/power plan; descriptive medians/p95/spreads, no significance claim.","- Long-time phase errors are characterized, not hidden behind the short-horizon regression threshold.","- High-precision elliptic and RK4 checks are independent of production descent. Double RK4 cannot be the oracle at the nextafter-pi separatrix limit.","- No production source changes or database/network calls. Frozen source copies and exact hashes accompany every result.","- No dedicated memory claim: this scalar routine allocates short AGM lists, and latency/accuracy dominate this comparison."]
 if summary["violations"]:lines += ["","Unexpected fixed violations:",""]+["- "+json.dumps(x,ensure_ascii=False) for x in summary["violations"][:10]]
 lines += ["", "Public jacobi_cn compatibility: %d deterministic inputs, %d bitwise differences versus baseline."%(compat["cases"],compat["bitwise_mismatches"]), "", "Public pendulum_truth period: %d independent high-precision comparisons, %d failed; max relative error %.6g and max absolute error %.6g s. The reference uses the declared binary64 theta0_rad; exact-real degree conversion differences near the separatrix are separately retained in period_truth_raw.json."%(periods["cases"],periods["failed_cases"],periods["fixed_max_relative_error"],periods["fixed_max_abs_error_s"])]
 (ROOT/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8",newline="\n");print(json.dumps({"accuracy_summary":{k:v for k,v in summary.items() if k!="violations"},"period_truth_summary":{k:v for k,v in periods.items() if k not in ("rows","reference_convention")},"public_cn_bitwise_mismatches":compat["bitwise_mismatches"],"rk4_fine_max_error":rk["fine_max_abs_error_rad"],"domain_failures":sum(not x["passed"] for x in dc),"report":str(ROOT/"report.md")},ensure_ascii=False),flush=True)
 return int(bool(summary["fixed_violations"] or any(not x["passed"] for x in dc) or compat["bitwise_mismatches"] or periods["failed_cases"]))
if __name__=="__main__":raise SystemExit(main())
