from astroquery.mast import Observations
import argparse


def jwst_query_mast(
    instrument_name=[],
    proposal_id=[],
    object_name=[],
    calib_levels=[1],
    filters=[],
    save_folder="./",
    curl_flag=False,
):
    """Get data from MAST archive
    Args:
        instrument_name : str : instrument name (e.g. NIRCAM, MIRI)
        proposal_id : int : proposal id (e.g. 1288 for ERS program 1288)
        filter : arr : filters name (F770W, MRS, ...)
        calib_level : arr : Calibration level:
                            0 = raw,
                            1 = uncalibrated,
                            2 = calibrated,
                            3 = science product,
                            4 = contributed science product
        object_name : str : object name (e.g. 'Cartwheel')
        save_folder : str : Folder for downloaded data (default = ./)
        curl_flag : bool : get link to download files later (default is False)
    Returns:
        dwld : Data files downloaded into save_folder
    """
    obs = Observations.query_criteria(
        obs_collection="JWST",
        instrument_name=instrument_name,
        proposal_id=proposal_id,
        objectname=object_name,
        calib_level=3,
        filters=filters,
    )
    data_products = Observations.get_product_list(obs)
    products = Observations.filter_products(
        data_products,
        productType=["SCIENCE"],
        extension="fits",
        calib_level=calib_levels,
    )

    print(
        """-- Found %i FITS files --
    Instrument = %s
    Object = %s
    Proposal ID = %s
    Calibration level(s) = %s
    Filter(s) = %s
    Output directory = '%s' \n"""
        % (
            len(products),
            instrument_name,
            obs["target_name"][0].replace("NAME ", ""),
            obs["proposal_id"][0],
            calib_levels,
            list(dict.fromkeys(list(obs["filters"]))),
            save_folder,
        )
    )
    print("Do you want to download the files ? Press y or n")
    download = input()
    if str(download) == "y":
        dwld = Observations.download_products(
            products,
            productType="SCIENCE",
            curl_flag=curl_flag,
            download_dir=save_folder,
        )
    return dwld


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument")
    ap.add_argument("--object")
    ap.add_argument("--proposal_id")
    ap.add_argument("--stage", nargs="+", type=int, default=0)
    args = ap.parse_args()
    option_kwarg_pairs = (
        (args.instrument, "instrument_name"),
        (args.object, "object_name"),
        (args.proposal_id, "proposal_id"),
        (args.stage, "calib_levels"),
    )
    kwargs = {
        kwarg_name: option_value
        for (option_value, kwarg_name) in option_kwarg_pairs
        if option_value is not None
    }
    jwst_query_mast(**kwargs)
