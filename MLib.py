import matplotlib.pyplot
import pandas
import numpy
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import pathlib
import urllib.request

CURRENT_OUTPUT_FOLDER_PATH : pathlib.Path
def InitPlot(OutPutImagesFolderPath : pathlib.Path = pathlib.Path() / "DefaultImagesFolder") -> None:
    matplotlib.pyplot.rc('font', size=12)
    matplotlib.pyplot.rc('axes', labelsize=14, titlesize=14)
    matplotlib.pyplot.rc('legend', fontsize=12)
    matplotlib.pyplot.rc('xtick', labelsize=10)
    matplotlib.pyplot.rc('ytick', labelsize=10)
    SetOutPutImagesFolderPath(OutPutImagesFolderPath)
    return

def SetOutPutImagesFolderPath(OutPutImagesFolderPath : pathlib.Path) -> None:
    global CURRENT_OUTPUT_FOLDER_PATH
    CURRENT_OUTPUT_FOLDER_PATH = OutPutImagesFolderPath
    CURRENT_OUTPUT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    return

def SaveFig(FileName : str, tight_layout : bool =True, fig_extension : str ="png", DPIresolution : int =300) -> None:
    path = CURRENT_OUTPUT_FOLDER_PATH / f"{FileName}.{fig_extension}"
    if tight_layout:
        matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(path, format=fig_extension, dpi=DPIresolution)
    return

def SaveFigAndShow(FileName : str, tight_layout : bool =True, fig_extension : str ="png", DPIresolution : int =300) -> None:
    SaveFig(FileName, tight_layout, fig_extension, DPIresolution)
    matplotlib.pyplot.show()
    return

def CSV2DataFrame(FilePath : str) -> pandas.DataFrame:
    return pandas.read_csv(FilePath)

def RequestAndSaveCSVFromURL(URL, FolderPath : pathlib.Path, FileName : pathlib.Path) -> None:
    FolderPath.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, FolderPath / FileName)
    return

def GetValuesFromDataFrameAsNumpyNdarray(DataFrame : pandas.DataFrame, Values : list) -> numpy.ndarray:
    return DataFrame[Values].to_numpy().reshape(-len(Values), len(Values))


InitPlot(pathlib.Path() / "images")
# Download and prepare the data
LifeSatCSVPath = "https://github.com/ageron/data/raw/main/lifesat/lifesat.csv"
DFLifeSat = CSV2DataFrame(LifeSatCSVPath)
print(DFLifeSat)
print('------------------------------------------------------------------------------------------------------------')

folderpath = pathlib.Path() / "dataset"
filename = "lifesat.csv"
RequestAndSaveCSVFromURL(LifeSatCSVPath, folderpath, filename)
df = CSV2DataFrame(folderpath / filename)
print(df)


# GDPLabel : str = "GDP per capita (USD)"
# LifeSatLabel : str = "Life satisfaction"
# # you can do it like this
# """
# array = GetValuesFromDataFrameAsNumpyNdarray(DFLifeSat, [GDPLabel, LifeSatLabel])
# X = array[:, 0].reshape(-1, 1)
# y = array[:, 1].reshape(-1, 1)
# """
# # or like this
# X = GetValuesFromDataFrameAsNumpyNdarray(DFLifeSat, [GDPLabel])
# y = GetValuesFromDataFrameAsNumpyNdarray(DFLifeSat, [LifeSatLabel])

# # Visualize the data
# DFLifeSat.plot(kind='scatter', grid=True, x=GDPLabel, y=LifeSatLabel)
# matplotlib.pyplot.axis([23_500, 62_500, 4, 9])
# SaveFigAndShow("plot")


# model = LinearRegression()
# model.fit(X, y)
# X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020
# print(model.predict(X_new)) # outputs [[6.30165767]]

# model = KNeighborsRegressor(n_neighbors=3)
# model.fit(X, y)
# # Make a prediction for Cyprus
# print(model.predict(X_new)) # outputs [[6.33333333]]