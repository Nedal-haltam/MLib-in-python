import matplotlib.pyplot
import pandas
import numpy
import scipy.stats
import sklearn
import sklearn.cluster
import sklearn.compose
import sklearn.ensemble
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
import joblib
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
import scipy
import sklearn.svm


class CalculationType(enum.Enum):
    enumMean = enum.auto()
    enumMedian = enum.auto()
    enumMode = enum.auto()
class MLALGORITHMTYPE(enum.Enum):
    enumlinearregression = enum.auto()
    enumdecisiontreeregression = enum.auto()
    enumrandomforestregression = enum.auto()
    enumsvr = enum.auto()

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

def DumpModelInFile(Model, FilePath : pathlib.Path):
    joblib.dump(Model, FilePath)

def LoadModelFromFile(FilePath : pathlib.Path):
    return joblib.load(FilePath)

def DropingOutLiers_testtest():
    # droping outliers
    # from sklearn.ensemble import IsolationForest
    # isolation_forest = IsolationForest(random_state=42)
    # outlier_pred = isolation_forest.fit_predict(ImputerOutPut)
    # DFsamples = DFsamples.iloc[outlier_pred == 1]
    # DFlabels = DFlabels.iloc[outlier_pred == 1]
    return

def GetFullPipeLine(PreProcessing : ColumnTransformer, MLAlgorithmType : MLALGORITHMTYPE) -> sklearn.pipeline.Pipeline:
    # TODO: refactor the names of the pipeline stages here into standalone variables so we can use them in other places without mistakes
    if MLAlgorithmType == MLALGORITHMTYPE.linearregression:
        return sklearn.pipeline.Pipeline([
            ("PreProc", PreProcessing),
            ("linear", sklearn.linear_model.LinearRegression()),
        ])
    if MLAlgorithmType == MLALGORITHMTYPE.decisiontreeregression:
        return sklearn.pipeline.Pipeline([
            ("PreProc", PreProcessing),
            ("decisiontree", sklearn.tree.DecisionTreeRegressor(random_state=42)),
        ])
    if MLAlgorithmType == MLALGORITHMTYPE.randomforestregression:
        return sklearn.pipeline.Pipeline([
            ("PreProc", PreProcessing),
            ("randomforest", sklearn.ensemble.RandomForestRegressor(random_state=42)),
        ])
    if MLAlgorithmType == MLALGORITHMTYPE.enumsvr:
        return sklearn.pipeline.Pipeline([
            ("PreProc", PreProcessing),
            ("svr", sklearn.svm.SVR()),
        ])        

def MLAlgorithms(PreProcessing : ColumnTransformer, DataFrameSamples : pandas.DataFrame, DataFrameLabels : pandas.DataFrame) -> None:
    # Linear Regressor
    FullPipeLine = GetFullPipeLine(PreProcessing, MLALGORITHMTYPE.linearregression)
    FullPipeLine.fit(DataFrameSamples, DataFrameLabels)
    DFsamplesPredictions = FullPipeLine.predict(DataFrameSamples)
    error = sklearn.metrics.root_mean_squared_error(DataFrameLabels, DFsamplesPredictions)
    print("Training Error of Linear Regressor : ", error)
    LinearRegressorCV = -sklearn.model_selection.cross_val_score(FullPipeLine, DataFrameSamples, DataFrameLabels, scoring="neg_root_mean_squared_error", cv=10)
    print("Validation Error statistics:\n", pandas.Series(LinearRegressorCV).describe())
    #########################################################################################################################################################
    # Decision Tree Regressor
    FullPipeLine = GetFullPipeLine(PreProcessing, MLALGORITHMTYPE.decisiontreeregression)
    FullPipeLine.fit(DataFrameSamples, DataFrameLabels)
    DFsamplesPredictions = FullPipeLine.predict(DataFrameSamples)
    error = sklearn.metrics.root_mean_squared_error(DataFrameLabels, DFsamplesPredictions)
    print("Training Error of Decision Tree Regressor : ", error)
    DecisionTreeRegressorCV = -sklearn.model_selection.cross_val_score(FullPipeLine, DataFrameSamples, DataFrameLabels, scoring="neg_root_mean_squared_error", cv=10)
    print("Validation Error statistics:\n", pandas.Series(DecisionTreeRegressorCV).describe())
    #########################################################################################################################################################
    # Random Forest Regressor
    FullPipeLine = GetFullPipeLine(PreProcessing, MLALGORITHMTYPE.randomforestregression)
    FullPipeLine.fit(DataFrameSamples, DataFrameLabels)
    DFsamplesPredictions = FullPipeLine.predict(DataFrameSamples)
    error = sklearn.metrics.root_mean_squared_error(DataFrameLabels, DFsamplesPredictions)
    print("Training Error of Random Forest Regressor : ", error)
    RandomForestRegressorCV = -sklearn.model_selection.cross_val_score(FullPipeLine, DataFrameSamples, DataFrameLabels, scoring="neg_root_mean_squared_error", cv=10)
    print("Validation Error statistics:\n", pandas.Series(RandomForestRegressorCV).describe())
    #########################################################################################################################################################
    return None

class ClusterSimilarity(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = sklearn.cluster.KMeans(self.n_clusters, n_init=10,
                              random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self  # always return self!

    def transform(self, X):
        return sklearn.metrics.pairwise.rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
    
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]

def ApplyColumnTransformerOnDataFrame(PreProcessing : ColumnTransformer, DataFrameSamples : pandas.DataFrame) -> pandas.DataFrame:
    ProcessedSamples = PreProcessing.fit_transform(DataFrameSamples)
    return pandas.DataFrame(ProcessedSamples, columns=PreProcessing.get_feature_names_out(), index=DataFrameSamples.index)

def ApplyGridSearchCV(PipeLine : ColumnTransformer, Parameters, DataFrameSamples : pandas.DataFrame, DataFrameLabels : pandas.DataFrame) -> sklearn.model_selection.GridSearchCV:
    GridSearch = sklearn.model_selection.GridSearchCV(PipeLine, Parameters, cv=3, scoring='neg_root_mean_squared_error')
    return GridSearch.fit(DataFrameSamples, DataFrameLabels)

def ApplyRandomSearchCV(PipeLine : ColumnTransformer, ParametersDistribution, DataFrameSamples : pandas.DataFrame, DataFrameLabels : pandas.DataFrame, Iterations : int) -> sklearn.model_selection.RandomizedSearchCV:
    RandomSearchCV = sklearn.model_selection.RandomizedSearchCV(PipeLine, param_distributions=ParametersDistribution, n_iter=Iterations, cv=3, scoring='neg_root_mean_squared_error', random_state=42)
    return RandomSearchCV.fit(DataFrameSamples, DataFrameLabels)

def RandomForestGridSearchAndRandomizedSearchCVs(preprocessing : ColumnTransformer, DFsamples : pandas.DataFrame, DFlabels : pandas.DataFrame, TestSet : pandas.DataFrame, OutPutLabel : str):
    FullPipeLine = GetFullPipeLine(preprocessing, MLALGORITHMTYPE.randomforestregression)
    # ParametersForRandomForest = [
    #     {'PreProc__geo__n_clusters': [5, 8, 10],
    #     'randomforest__max_features': [4, 6, 8]},
    #     {'PreProc__geo__n_clusters': [10, 15],
    #     'randomforest__max_features': [6, 8, 10]},
    # ]
    # GridSearch = ApplyGridSearchCV(FullPipeLine, Parameters, DFsamples, DFlabels)
    # print('Best set of parameters are : \n', GridSearch.best_params_)
    # print('Best Estimator : \n', GridSearch.best_estimator_)  # includes preprocessing
    # DFGridSearchCV = pandas.DataFrame(GridSearch.cv_results_)
    # DFGridSearchCV.sort_values(by="mean_test_score", ascending=False, inplace=True)
    # DFGridSearchCV.head()
    """
    how to choose the sampling distribution for a hyperparameter

    `scipy.stats.randint(a, b+1)`: for hyperparameters with _discrete_ values that range from a to b, and all values in that range seem equally likely.
    `scipy.stats.uniform(a, b)`: this is very similar, but for _continuous_ hyperparameters.
    `scipy.stats.geom(1 / scale)`: for discrete values, when you want to sample roughly in a given scale. E.g., with scale=1000 most samples will be in this ballpark, but ~10% of all samples will be <100 and ~10% will be >2300.
    `scipy.stats.expon(scale)`: this is the continuous equivalent of `geom`. Just set `scale` to the most likely value.
    `scipy.stats.loguniform(a, b)`: when you have almost no idea what the optimal hyperparameter value's scale is. If you set a=0.01 and b=100, then you're just as likely to sample a value between 0.01 and 0.1 as a value between 10 and 100.
    """
    ParametersDistribution = {'PreProc__geo__n_clusters': scipy.stats.randint(low=3, high=50), 'randomforest__max_features': scipy.stats.randint(low=2, high=20)}
    RandomSearchCV = ApplyRandomSearchCV(FullPipeLine, ParametersDistribution, DFsamples, DFlabels, 10)
    # print('Best set of parameters are : \n', RandomSearchCV.best_params_)
    # print('Best Estimator : \n', RandomSearchCV.best_estimator_)  # includes preprocessing
    # DFRandomSearchCV = pandas.DataFrame(RandomSearchCV.cv_results_)
    # DFRandomSearchCV.sort_values(by="mean_test_score", ascending=False, inplace=True)
    # DFRandomSearchCV.head()
    FinalModel : sklearn.pipeline.Pipeline = RandomSearchCV.best_estimator_  # includes preprocessing
    # FMrf : sklearn.ensemble.RandomForestRegressor = FinalModel["randomforest"]
    # FeatureImportance = FMrf.feature_importances_.round(2)
    # print(FeatureImportance)
    # print(sorted(zip(FeatureImportance, FinalModel["PreProc"].get_feature_names_out()), reverse=True))
    TestSetSamples = TestSet.drop(OutPutLabel, axis=1)
    TestSetLabels = TestSet[OutPutLabel].copy()
    FinalPredictions = FinalModel.predict(TestSetSamples)
    FinalError = sklearn.metrics.root_mean_squared_error(TestSetLabels, FinalPredictions)
    print(FinalError)


def PreProcessDataFrame(InputDataFrame : pandas.DataFrame, OutPutLabel : str) -> None:

    # see the following two lines before and after constructing some new features through combining current ones
    # CorrelationMatrix = InputDataFrame.corr(numeric_only=True)
    # CorrelationMatrix[OutPutLabel].sort_values(ascending=False)

    # drop the output column
    TrainSet, TestSet = sklearn.model_selection.train_test_split(InputDataFrame, test_size=0.2, random_state=42)
    TrainSet : pandas.DataFrame
    TestSet : pandas.DataFrame
    DFsamples : pandas.DataFrame = TrainSet.drop(OutPutLabel, axis=1)
    DFlabels : pandas.DataFrame = TrainSet[OutPutLabel].copy(True)
    # begin data cleaning
    # sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))
    ImputationStrategy : str = (CalculationType.enumMean).__str__().split('.')[1].lower().removeprefix('enum')
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
    # ClusterSimilarity_pipeline = sklearn.pipeline.Pipeline([
    #     ('clustering', ClusterSimilarity(n_clusters=10, gamma=1., random_state=42))
    # ])
    ClusterSimilarity_pipeline = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    # DropingOutLiers_testtest()
    # keep in mind that you should try all pipelines on all columns to you can shoose the best features that correlates with the output
    LogColumns = ["total_bedrooms", "total_rooms", "population", "households", "median_income"]
    # TODO: refactor the names of the pipeline stages here into standalone variables so we can use them in other places without mistakes
    preprocessing = ColumnTransformer([
        ("log", log_pipeline, LogColumns),
        ("geo", ClusterSimilarity_pipeline, ['longitude', 'latitude']),
        ("cat", cat_pipeline, CatigoricalSelector),
    ], remainder=num_pipeline)
    # DFsamplesFinal = ApplyColumnTransformerOnDataFrame(preprocessing, DFsamples)
    # print(DFsamplesFinal.describe().T)
    # print(pandas.concat([DFsamplesFinal, DFlabels], axis=1).corr(numeric_only=True)[OutPutLabel].sort_values(ascending=False))
    # MLAlgorithms(DFsamplesFinal, DFlabels)
    # ParametersFormat = 
    # [
    #     {
    #         'parameter': ["""list of possible values"""],
    #         'parameter': ["""list of possible values"""],
    #     },
    #     {
    #         'parameter': ["""list of possible values"""],
    #         'parameter': ["""list of possible values"""],
    #     },
    # ]
    RandomForestGridSearchAndRandomizedSearchCVs(preprocessing, DFsamples, DFlabels, TestSet, OutPutLabel)
    ParametersForSVR = [
        {
            'svr__kernel': ['linear'], 
            'svr__C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0],
        },
        {
            'svr__kernel': ['rbf'], 
            'svr__C': [1.0, 3.0, 10., 30., 100., 300., 1000.0], 
            'svr__gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
        },
    ]
    FullPipeLine = GetFullPipeLine(preprocessing, MLALGORITHMTYPE.enumsvr)
    GridSearch = ApplyGridSearchCV(FullPipeLine, ParametersForSVR, DFsamples, DFlabels)
    print('Best set of parameters are : \n', GridSearch.best_params_)
    print('Best Estimator : \n', GridSearch.best_estimator_)  # includes preprocessing
    DFGridSearchCV = pandas.DataFrame(GridSearch.cv_results_)
    DFGridSearchCV.sort_values(by="mean_test_score", ascending=False, inplace=True)
    DFGridSearchCV.head()

    ParametersDistribution = {
        'svr__kernel': ['linear', 'rbf'],
        'svr__C': scipy.stats.loguniform(20, 200_000),
        'svr__gamma': scipy.stats.expon(scale=1.0),
    }
    RandomSearchCV = ApplyRandomSearchCV(FullPipeLine, ParametersDistribution, DFsamples, DFlabels, 50)
    # print('Best set of parameters are : \n', RandomSearchCV.best_params_)
    # print('Best Estimator : \n', RandomSearchCV.best_estimator_)  # includes preprocessing
    # DFRandomSearchCV = pandas.DataFrame(RandomSearchCV.cv_results_)
    # DFRandomSearchCV.sort_values(by="mean_test_score", ascending=False, inplace=True)
    # DFRandomSearchCV.head()
    FinalModel : sklearn.pipeline.Pipeline = RandomSearchCV.best_estimator_  # includes preprocessing
    # FMrf : sklearn.ensemble.RandomForestRegressor = FinalModel["randomforest"]
    # FeatureImportance = FMrf.feature_importances_.round(2)
    # print(FeatureImportance)
    # print(sorted(zip(FeatureImportance, FinalModel["PreProc"].get_feature_names_out()), reverse=True))
    TestSetSamples = TestSet.drop(OutPutLabel, axis=1)
    TestSetLabels = TestSet[OutPutLabel].copy()
    FinalPredictions = FinalModel.predict(TestSetSamples)
    FinalError = sklearn.metrics.root_mean_squared_error(TestSetLabels, FinalPredictions)
    print(FinalError)

    return None


InitPlot(RESOURCES_FOLDER_PATH / "images")
PreProcessDataFrame(CSV2DataFrame(RESOURCES_FOLDER_PATH / "ds" / "housing.csv"), "median_house_value")

# LifeSatExample()
# model = LinearRegression()
# model.fit(X, y)
# X_new = [[37_655.2]]  # Cyprus' GDP per capita in 2020
# print(model.predict(X_new)) # outputs [[6.30165767]]

# model = KNeighborsRegressor(n_neighbors=3)
# model.fit(X, y)
# # Make a prediction for Cyprus
# print(model.predict(X_new)) # outputs [[6.33333333]]
