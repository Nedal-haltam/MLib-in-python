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

