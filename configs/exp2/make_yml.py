import yaml
import itertools
with open("drugs_schcome_scratch.yml","rb") as f:
    config = yaml.safe_load(f)

radials = [4, 8, 16]
sphericals = [4, 6, 8]
#seeds = [20220 + i for i in range(5)]
seeds = [2022]
idx = 0
for r, s in itertools.product(radials, sphericals):
    idx += 1
    for seed in seeds:
        config["train"]["seed"] = seed
        config["model"]["num_radial"] = r
        config["model"]["num_spherical"] = s


    with open(f"drugs_exp2_{idx}.yml","w") as f:
        yaml.dump(config, f)


with open("ts_schcome_scratch.yml","rb") as f:
    config = yaml.safe_load(f)

radials = [4, 8, 16]
sphericals = [4, 6, 8]
#seeds = [20220 + i for i in range(5)]
seeds = [2022]
idx = 0
for r, s in itertools.product(radials, sphericals):
    idx += 1
    for seed in seeds:
        config["train"]["seed"] = seed
        config["model"]["num_radial"] = r
        config["model"]["num_spherical"] = s


    with open(f"ts_exp2_{idx}.yml","w") as f:
        yaml.dump(config, f)
