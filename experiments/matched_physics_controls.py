"""Matched controller study: plasticity, simple heat pulse, capacitor, and RLC."""
from experiments.long_regimes import run

def study(seed=17, updates=16):
    sequence=["A","B","A","B","A"]
    kinds=("none","plastic","pulse","cap","rlc")
    results={kind:run(kind,seed,sequence,updates) for kind in kinds}
    base=results["none"]
    return {kind:{"regret":r["current_regret_proxy"],"final_A":r["trajectory"][-1]["A"],"regret_gain":0.0 if kind=="none" else 1-r["current_regret_proxy"]/base["current_regret_proxy"],"final_A_gain":0.0 if kind=="none" else 1-r["trajectory"][-1]["A"]/base["trajectory"][-1]["A"],"max_temperature":r["max_temperature"],"injection":r["injection"]} for kind,r in results.items()}

if __name__=="__main__":
    import json
    print(json.dumps(study(),indent=2,sort_keys=True))
