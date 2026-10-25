import json, sim2
json.dump({"conv": sim2.exp_convergence(), "sens": sim2.exp_sensitivity()}, open("results_A.json", "w"), default=float)
