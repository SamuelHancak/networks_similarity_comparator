# import matplotlib.pyplot as plt
# import pandas as pd
# from modules.DataNormalizer import DataNormalizer

# from shapely import MultiPoint
# from shapely.geometry import Polygon

# input = (
#     DataNormalizer(
#         pd.read_csv("output/graphlet_counts.csv")
#     ).percentual_normalization()["geo1k_4k"]
#     * 100
# )

# coordinates = input.values.reshape((15, 2))


# ob = Polygon(coordinates)

# x, y = ob.exterior.xy

# plt.plot(x, y, color="b", alpha=0.7, linewidth=2, solid_capstyle="round", zorder=2)
# plt.fill(x, y, color="b", alpha=0.3, zorder=1)

# plt.title("Polygon Display using Shapely and Matplotlib")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")

# plt.show()

import pandas as pd
from shapely.geometry import Polygon
from modules.DataNormalizer import DataNormalizer

df = (
    DataNormalizer(
        pd.read_csv("output/graphlet_counts.csv").drop(columns=["Unnamed: 0"])
    ).percentual_normalization()
    * 100
)

polygons_arr = []
for col in df.columns:
    coordinates = df[col].values.reshape((15, 2))

    poly = Polygon(coordinates)
    poly = poly.buffer(0)
    polygons_arr.append(poly)

print(polygons_arr[2].intersection(polygons_arr[1]).area)
print(polygons_arr[2].union(polygons_arr[1]).area)
print(
    polygons_arr[2].intersection(polygons_arr[1]).area
    / polygons_arr[2].union(polygons_arr[1]).area
)
