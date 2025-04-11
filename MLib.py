import matplotlib.pyplot
import pandas
import numpy
import sklearn
import sklearn.compose
import sklearn.impute
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.neighbors import KNeighborsRegressor
import pathlib
import urllib.request
import urllib.request
import copy
import enum
from sklearn.compose import ColumnTransformer
import sklearn.pipeline
import sklearn.preprocessing
import threading

class CalculationType(enum.Enum):
    Mean = enum.auto()
    Median = enum.auto()
    Mode = enum.auto()

RESOURCES_FOLDER_PATH : pathlib.Path = pathlib.Path() / "resources"
CURRENT_OUTPUT_FOLDER_PATH : pathlib.Path = RESOURCES_FOLDER_PATH / "DefaultImagesFolder"
def InitPlot(OutPutImagesFolderPath : pathlib.Path = pathlib.Path() / "DefaultImagesFolder") -> None:
    matplotlib.pyplot.rc('font', size=12)
    matplotlib.pyplot.rc('axes', labelsize=14, titlesize=14)
    matplotlib.pyplot.rc('legend', fontsize=12)
    matplotlib.pyplot.rc('xtick', labelsize=10)
    matplotlib.pyplot.rc('ytick', labelsize=10)
    SetOutPutImagesFolderPath(OutPutImagesFolderPath)
    return None

def SetOutPutImagesFolderPath(OutPutImagesFolderPath : pathlib.Path) -> None:
    global CURRENT_OUTPUT_FOLDER_PATH
    CURRENT_OUTPUT_FOLDER_PATH = OutPutImagesFolderPath
    CURRENT_OUTPUT_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
    return None

def SaveFig(FileName : str, tight_layout : bool =True, fig_extension : str ="png", DPIresolution : int =300) -> None:
    path = CURRENT_OUTPUT_FOLDER_PATH / f"{FileName}.{fig_extension}"
    if tight_layout:
        matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(path, format=fig_extension, dpi=DPIresolution)
    return None

def SaveFigAndShow(FileName : str, tight_layout : bool =True, fig_extension : str ="png", DPIresolution : int =300) -> None:
    SaveFig(FileName, tight_layout, fig_extension, DPIresolution)
    matplotlib.pyplot.show()
    return None

def CSV2DataFrame(FilePath : str) -> pandas.DataFrame:
    return pandas.read_csv(FilePath)

def RequestAndSaveCSVFromURL(URL, FolderPath : pathlib.Path, FileName : str) -> None:
    FolderPath.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, FolderPath / FileName)
    return None

def GetValuesFromDataFrameAsNumpyNdarray(DataFrame : pandas.DataFrame, Values : list) -> numpy.ndarray:
    return DataFrame[Values].to_numpy().reshape(-len(Values), len(Values))

# def LifeSatExample() -> None:
    data_root = "https://github.com/ageron/data/raw/main/"
    DFLifeSat = pandas.read_csv(data_root + "lifesat/lifesat.csv")
    DFLifeSat.set_index("Country", inplace=True)
    DFLifeSat.sort_values(by='Country', inplace=True)
    # print(DFLifeSat.head())
    GDPLabel : str = "GDP per capita (USD)"
    LifeSatLabel : str = "Life satisfaction"
    # you can do it like this
    """
    array = GetValuesFromDataFrameAsNumpyNdarray(DFLifeSat, [GDPLabel, LifeSatLabel])
    X = array[:, 0].reshape(-1, 1)
    y = array[:, 1].reshape(-1, 1)
    """
    # or like this
    X = GetValuesFromDataFrameAsNumpyNdarray(DFLifeSat, [GDPLabel])
    y = GetValuesFromDataFrameAsNumpyNdarray(DFLifeSat, [LifeSatLabel])
    # Visualize the data
    DFLifeSat.plot(kind='scatter', grid=True, x=GDPLabel, y=LifeSatLabel)
    matplotlib.pyplot.axis([23_500, 62_500, 4, 9])
    matplotlib.pyplot.show()
    return None

def shuffle_and_split_data(InputDataFrame : pandas.DataFrame, test_ratio : int) -> pandas.DataFrame:
    shuffled_indices = numpy.random.permutation(len(InputDataFrame))
    test_set_size = int(len(InputDataFrame) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return InputDataFrame.iloc[train_indices], InputDataFrame.iloc[test_indices]

def Separator() -> None:
    print('-' * 150)
    return None

def DisplayDataFrameInfo(InputDataFrame : pandas.DataFrame, OutPutLabel : str) -> None:
    print(InputDataFrame.head())
    Separator()
    InputDataFrame.info()
    Separator()
    for col in InputDataFrame.columns:
        if InputDataFrame[col].dtype == object:
            # print("\t\t" , col)
            print(InputDataFrame[col].value_counts())
    Separator()    
    print(InputDataFrame.describe().T)
    Separator()
    print(InputDataFrame.isnull().sum())
    Separator()
    # see correlation with the output
    CorrelationMatrix = InputDataFrame.corr(numeric_only=True)
    CorrelationMatrix[OutPutLabel].sort_values(ascending=False)

    return None

def VisualizeDataFrame(InputDataFrame : pandas.DataFrame) -> None:
    # done or modified later using other tools katkpt
    InputDataFrame.hist(bins=50, figsize=(12, 8))
    SaveFig("histogram")
    pandas.plotting.scatter_matrix(InputDataFrame, figsize=(12, 8))
    SaveFig("scatter_matrix_plot")
    return None

def ApplyPiplineOnDataFrame(pipeline : sklearn.pipeline.Pipeline, dataframe : pandas.DataFrame) -> pandas.DataFrame:
    samples_num_prepared = pipeline.fit_transform(dataframe)
    return pandas.DataFrame( samples_num_prepared, columns=pipeline.get_feature_names_out(), index=dataframe.index)


def PreProcessDataFrame(InputDataFrame : pandas.DataFrame, OutPutLabel : str, CalcType : CalculationType) -> None:

    # see the following two lines before and after constructing some new features through combining current ones
    # CorrelationMatrix = InputDataFrame.corr(numeric_only=True)
    # CorrelationMatrix[OutPutLabel].sort_values(ascending=False)

    # drop the output column
    DFsamples = InputDataFrame.drop(OutPutLabel, axis=1)
    DFlabels = InputDataFrame[OutPutLabel].copy(True)
    # begin data cleaning
    # sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    ImputationStrategy : str = CalcType.__str__().split('.')[1].lower()
    NumericalSelector = sklearn.compose.make_column_selector(dtype_include=numpy.number)
    CatigoricalSelector = sklearn.compose.make_column_selector(dtype_include=object)
    cat_pipeline = sklearn.pipeline.Pipeline([
        ("impute", sklearn.impute.SimpleImputer(strategy="most_frequent")),
        ("encode", sklearn.preprocessing.OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])
    num_pipeline = sklearn.pipeline.Pipeline([
        ("impute", sklearn.impute.SimpleImputer(strategy=ImputationStrategy)),
        # ("standarize", sklearn.preprocessing.StandardScaler()),
        ("standarize", sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))),
    ])
    log_pipeline = sklearn.pipeline.Pipeline([
        ("impute", sklearn.impute.SimpleImputer(strategy=ImputationStrategy)),
        ("log", sklearn.preprocessing.FunctionTransformer(numpy.log, feature_names_out="one-to-one")),
        # ("standarize", sklearn.preprocessing.StandardScaler()),
        ("standarize", sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))),
    ])

    LogColumns = ["total_bedrooms", "total_rooms", "population", "households", "median_income"]
    preprocessing = ColumnTransformer([
        ("log", log_pipeline, LogColumns),
        ("num", num_pipeline, NumericalSelector),
        ("cat", cat_pipeline, CatigoricalSelector),
    ], remainder=num_pipeline)
    
    # Linear Regressor
    LinearRegressorPL = sklearn.pipeline.Pipeline([
        ("PreProc", preprocessing),
        ("LinearRegressor", sklearn.linear_model.LinearRegression()),
    ])
    LinearRegressorPL.fit(DFsamples, DFlabels)
    DFsamplesPredictions = LinearRegressorPL.predict(DFsamples)
    error = sklearn.metrics.root_mean_squared_error(DFlabels, DFsamplesPredictions)
    print("Error of Linear Regressor : ", error)
    LinearRegressorCV = -sklearn.model_selection.cross_val_score(LinearRegressorPL, DFsamples, DFlabels, scoring="neg_root_mean_squared_error", cv=10)
    print(pandas.Series(LinearRegressorCV).describe())

    # ProcessedSamples = preprocessing.fit_transform(DFsamples)
    # DFsamplesFinal = pandas.DataFrame(ProcessedSamples, columns=preprocessing.get_feature_names_out(), index=DFsamples.index)
    # print(DFsamplesFinal.describe().T)
    # print(pandas.concat([DFsamplesFinal, DFlabels], axis=1).corr(numeric_only=True)[OutPutLabel].sort_values(ascending=False))
    # droping outliers
    # from sklearn.ensemble import IsolationForest
    # isolation_forest = IsolationForest(random_state=42)
    # outlier_pred = isolation_forest.fit_predict(ImputerOutPut)
    # DFsamples = DFsamples.iloc[outlier_pred == 1]
    # DFlabels = DFlabels.iloc[outlier_pred == 1]




    # train_set, test_set = sklearn.model_selection.train_test_split(DFHousing, test_size=0.2, random_state=42)
    return None


InitPlot(RESOURCES_FOLDER_PATH / "images")
DFHousing = CSV2DataFrame(RESOURCES_FOLDER_PATH / "ds" / "housing.csv")

outputlabel = "median_house_value"
PreProcessDataFrame(DFHousing, outputlabel, CalculationType.Median)

# LifeSatExample()
# model = LinearRegression()
# model.fit(X, y)
# X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020
# print(model.predict(X_new)) # outputs [[6.30165767]]

# model = KNeighborsRegressor(n_neighbors=3)
# model.fit(X, y)
# # Make a prediction for Cyprus
# print(model.predict(X_new)) # outputs [[6.33333333]]
