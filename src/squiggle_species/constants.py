SPECIES = ("LB01", "LB06", "LB07", "LB08", "LB09", "LB12", "LB18", "LB11", "LB02")
SPECIES_TO_LABEL = {name: index for index, name in enumerate(SPECIES)}
LABEL_TO_SPECIES = {index: name for name, index in SPECIES_TO_LABEL.items()}
SPLITS = ("train", "val", "test")

