import zarr

path = "/bsoden3/ewellmeyer/CMIP6/by_model/CESM2/original_hist/cesm2LE-historical-cmip6-PRECL.zarr"
root = zarr.open_group(path, mode="r")

print("array_keys:", list(root.array_keys()))
print("group_keys:", list(root.group_keys()))

arr = root["PRECL"]
print("PRECL shape:", arr.shape)
print("PRECL chunks:", arr.chunks)
print("PRECL compressor:", arr.compressor)
