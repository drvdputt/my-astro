# load csv table downloaded from mast (the export table button on the search reaults)
# then download all files based on the obsid.
import numpy as np
from astroquery.mast import Observations
from astropy.table import Table
from sys import argv

print("opening ", argv[1])
# obsid_table = Table.read('/Users/dvandeputte/Downloads/Proposal_IDs_1288(1).csv')
obsid_table = Table.read(argv[1], comment="#")

# custom filter
keep = np.array(["c1002" in row["obs_id"] for row in obsid_table])

obsids = obsid_table[keep]["obs_id"]
allobs = Observations.query_criteria(obs_id=obsids)

for obs in allobs:
    print(obs)
    products = Observations.get_product_list(obs)
    fits_products = Observations.filter_products(
        products, productType=["SCIENCE"], extension="fits", calib_level=[1]
    )
    print(fits_products)

    if argv[2] == "y":
        answer = "y"
    else:
        answer = input("download these? y/n")

    if answer.lower() == "y":
        Observations.download_products(fits_products)
