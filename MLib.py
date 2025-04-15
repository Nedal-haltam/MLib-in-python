import urllib.request
import matplotlib
import matplotlib.pyplot
import pandas
import numpy
import scipy.stats
import sklearn
import pathlib
import urllib
import enum
import joblib
import scipy
import sklearn.compose
import sklearn.datasets
import sklearn.ensemble
import sklearn.impute
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm
import sklearn.tree

# TODO: utilize these
# import sklearn.feature_extraction
# import sklearn.feature_selection
# from sklearn.experimental import enable_halving_search_cv
# from sklearn.model_selection.cross_val_score HalvingRandomSearchCV
# TODO: investigate in the different parameters of the functions we use in the function calls
class CalculationType(enum.Enum):
    enumMean = enum.auto()
    enumMedian = enum.auto()
    enumMode = enum.auto()
class MLALGORITHMTYPE(enum.Enum):
    enumlinearregression = enum.auto()
    enumdecisiontreeregression = enum.auto()
    enumrandomforestregression = enum.auto()
    enumsvr = enum.auto()

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
    [print(InputDataFrame[col].value_counts()) for col in InputDataFrame.select_dtypes(include='object').columns]
    Separator()    
    print(InputDataFrame.describe().T)
    Separator()
    print(InputDataFrame.isnull().sum())
    Separator()
    # see correlation with the output
    if OutPutLabel in InputDataFrame.columns:
        print(InputDataFrame.corr(numeric_only=True)[OutPutLabel].sort_values(ascending=False))

    return None

def VisualizeDataFrame(InputDataFrame : pandas.DataFrame) -> None:
    # TODO: done or modified later using other tools katkpt
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

# TODO: remove outliers if there is (if applicable)
def DropingOutLiers_testtest():
    # droping outliers
    # from sklearn.ensemble import IsolationForest
    # isolation_forest = IsolationForest(random_state=42)
    # outlier_pred = isolation_forest.fit_predict(ImputerOutPut)
    # DFsamples = DFsamples.iloc[outlier_pred == 1]
    # DFlabels = DFlabels.iloc[outlier_pred == 1]
    return

def GetFullPipeLine(PreProcessing : sklearn.compose.ColumnTransformer, MLAlgorithmType : MLALGORITHMTYPE) -> sklearn.pipeline.Pipeline:
    # TODO: refactor the names of the pipeline stages here into standalone variables so we can use them in other places without mistakes
    if MLAlgorithmType == MLALGORITHMTYPE.enumlinearregression:
        return sklearn.pipeline.Pipeline([
            ("PreProc", PreProcessing),
            ("linear", sklearn.linear_model.LinearRegression()),
        ])
    if MLAlgorithmType == MLALGORITHMTYPE.enumdecisiontreeregression:
        return sklearn.pipeline.Pipeline([
            ("PreProc", PreProcessing),
            ("decisiontree", sklearn.tree.DecisionTreeRegressor(random_state=42)),
        ])
    if MLAlgorithmType == MLALGORITHMTYPE.enumrandomforestregression:
        return sklearn.pipeline.Pipeline([
            ("PreProc", PreProcessing),
            ("randomforest", sklearn.ensemble.RandomForestRegressor(random_state=42)),
        ])
    if MLAlgorithmType == MLALGORITHMTYPE.enumsvr:
        return sklearn.pipeline.Pipeline([
            ("PreProc", PreProcessing),
            ("svr", sklearn.svm.SVR()),
        ])        

def MLAlgorithmsOnNormalCV(PreProcessing : sklearn.compose.ColumnTransformer, DataFrameSamples : pandas.DataFrame, DataFrameLabels : pandas.DataFrame, MLAlgorithmType : MLALGORITHMTYPE) -> None:
    cv = 3
    FullPipeLine = GetFullPipeLine(PreProcessing, MLAlgorithmType)
    FullPipeLine.fit(DataFrameSamples, DataFrameLabels)
    DFsamplesPredictions = FullPipeLine.predict(DataFrameSamples)
    error = sklearn.metrics.root_mean_squared_error(DataFrameLabels, DFsamplesPredictions)
    print(f"Training Error of `{MLAlgorithmType.__str__().split('.')[1].lower().removeprefix('enum')}` : ", error)
    CVresult = -sklearn.model_selection.cross_val_score(FullPipeLine, DataFrameSamples, DataFrameLabels, scoring='neg_root_mean_squared_error', cv=cv)
    print("Cross Validation statistics:\n", pandas.Series(CVresult).describe())
    return None

def ApplyColumnTransformerOnDataFrame(PreProcessing : sklearn.compose.ColumnTransformer, DataFrameSamples : pandas.DataFrame) -> pandas.DataFrame:
    ProcessedSamples = PreProcessing.fit_transform(DataFrameSamples)
    return pandas.DataFrame(ProcessedSamples, columns=PreProcessing.get_feature_names_out(), index=DataFrameSamples.index)

def ApplyGridSearchCV(PipeLine, Parameters, DataFrameSamples : pandas.DataFrame, DataFrameLabels : pandas.DataFrame, CVFolds : int) -> sklearn.model_selection.GridSearchCV:
    GridSearch = sklearn.model_selection.GridSearchCV(PipeLine, Parameters, cv=CVFolds, scoring='neg_root_mean_squared_error')
    return GridSearch.fit(DataFrameSamples, DataFrameLabels)

def ApplyRandomSearchCV(PipeLine, ParametersDistribution, DataFrameSamples : pandas.DataFrame, DataFrameLabels : pandas.DataFrame, Iterations : int, CVFolds : int) -> sklearn.model_selection.RandomizedSearchCV:
    RandomSearchCV = sklearn.model_selection.RandomizedSearchCV(PipeLine, param_distributions=ParametersDistribution, n_iter=Iterations, cv=CVFolds, scoring='neg_root_mean_squared_error', random_state=42)
    return RandomSearchCV.fit(DataFrameSamples, DataFrameLabels)

def GridRandom_SearchCV(preprocessing : sklearn.compose.ColumnTransformer, Parameters, ParametersDistribution, Iterations, CVFolds, MLAlgorithmType : MLALGORITHMTYPE, DFsamples : pandas.DataFrame, DFlabels : pandas.DataFrame, TestSet : pandas.DataFrame, OutPutLabel : str):
    FullPipeLine = GetFullPipeLine(preprocessing, MLAlgorithmType)
    GridSearchCV = ApplyGridSearchCV(FullPipeLine, Parameters, DFsamples, DFlabels, CVFolds)
    DisplayCVInformation(GridSearchCV)
    #########################################################################################################################################################
    RandomSearchCV = ApplyRandomSearchCV(FullPipeLine, ParametersDistribution, DFsamples, DFlabels, Iterations ,CVFolds)
    DisplayCVInformation(RandomSearchCV)
    #########################################################################################################################################################
    # Test_BestEstimator_On_A_Set((CrossValidationObject).best_estimator_)
    # FMrf : sklearn.ensemble.RandomForestRegressor = FinalModel["randomforest"]
    # FeatureImportance = FMrf.feature_importances_.round(2)
    # print(FeatureImportance)
    # print(sorted(zip(FeatureImportance, FinalModel["PreProc"].get_feature_names_out()), reverse=True))

def GetError_On_A_Set(Model : sklearn.pipeline.Pipeline, Set : pandas.DataFrame, OutPutLabel : str):
    SetSamples = Set.drop(OutPutLabel, axis=1)
    SetLabels = Set[OutPutLabel].copy()
    FinalPredictions = Model.predict(SetSamples)
    return sklearn.metrics.root_mean_squared_error(SetLabels, FinalPredictions)

def DisplayFeaturesAndOutPutCorrelation(samples, labels, OutPutLabel):
    print(pandas.concat([samples, labels], axis=1).corr(numeric_only=True)[OutPutLabel].sort_values(ascending=False))

def DisplayCVInformation(CrossValidator):
    print(f'Best Score (Error) : {-CrossValidator.best_score_}')
    print('Best set of parameters are : \n', CrossValidator.best_params_)
    SVR_DFRandomSearchCV = pandas.DataFrame(CrossValidator.cv_results_)
    SVR_DFRandomSearchCV.sort_values(by="mean_test_score", ascending=False, inplace=True)
    print(SVR_DFRandomSearchCV.head())


def HousingExampleRegression(InputDataFrame : pandas.DataFrame, OutPutLabel : str) -> None:

    # see the following two lines before and after constructing some new features through combining current ones
    # CorrelationMatrix = InputDataFrame.corr(numeric_only=True)
    # CorrelationMatrix[OutPutLabel].sort_values(ascending=False)

    # drop the output column
    TrainSet, TestSet = sklearn.model_selection.train_test_split(InputDataFrame, test_size=0.2, random_state=42)
    TrainSet : pandas.DataFrame
    TrainSetsamples : pandas.DataFrame = TrainSet.drop(OutPutLabel, axis=1)
    TrainSetlabels : pandas.DataFrame = TrainSet[OutPutLabel].copy(True)
    TestSet : pandas.DataFrame
    TestSetsamples : pandas.DataFrame = TestSet.drop(OutPutLabel, axis=1)
    TestSetlabels : pandas.DataFrame = TestSet[OutPutLabel].copy(True)
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
    # keep in mind that you should try all pipelines on all columns to you can shoose the best features that correlates with the output
    LogColumns = ["total_bedrooms", "total_rooms", "population", "households", "median_income"]
    """
    how to choose the sampling distribution for a hyperparameter
    `scipy.stats.randint(a, b+1)`: for hyperparameters with _discrete_ values that range from a to b, and all values in that range seem equally likely.
    `scipy.stats.uniform(a, b)`: this is very similar, but for _continuous_ hyperparameters.
    `scipy.stats.geom(1 / scale)`: for discrete values, when you want to sample roughly in a given scale. E.g., with scale=1000 most samples will be in this ballpark, but ~10% of all samples will be <100 and ~10% will be >2300.
    `scipy.stats.expon(scale)`: this is the continuous equivalent of `geom`. Just set `scale` to the most likely value.
    `scipy.stats.loguniform(a, b)`: when you have almost no idea what the optimal hyperparameter value's scale is. If you set a=0.01 and b=100, then you're just as likely to sample a value between 0.01 and 0.1 as a value between 10 and 100.
    """
    # TODO: refactor the names of the pipeline stages here into standalone variables so we can use them in other places without mistakes
    preprocessing = sklearn.compose.ColumnTransformer([
        ("log", log_pipeline, LogColumns),
        ("geo", ClusterSimilarity_pipeline, ['longitude', 'latitude']),
        ("cat", cat_pipeline, CatigoricalSelector),
    ], remainder=num_pipeline)
    # a paramter is a list of dicts each consists of bunch of these ('parameter': ["""list of possible values"""], `the other one`)
    ParametersDistributionForRandomForest = {'PreProc__geo__n_clusters': scipy.stats.randint(low=3, high=50), 'randomforest__max_features': scipy.stats.randint(low=2, high=20)}
    ParametersForRandomForest = [
        {'PreProc__geo__n_clusters': [5, 8, 10],
        'randomforest__max_features': [4, 6, 8]},
        {'PreProc__geo__n_clusters': [10, 15],
        'randomforest__max_features': [6, 8, 10]},
    ]
    GridRandom_SearchCV(preprocessing, ParametersForRandomForest, ParametersDistributionForRandomForest, 3, 2, MLALGORITHMTYPE.enumrandomforestregression, TrainSetsamples, TrainSetlabels, TestSet, OutPutLabel)
    exit()
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
    ParametersDistributionForSVR = {
        'svr__kernel': ['linear', 'rbf'],
        'svr__C': scipy.stats.loguniform(20, 200_000),
        'svr__gamma': scipy.stats.expon(scale=1.0),
    }
    GridRandom_SearchCV(preprocessing, ParametersForSVR, ParametersDistributionForSVR, 3, 2, MLALGORITHMTYPE.enumsvr, TrainSetsamples, TrainSetlabels, TestSet, OutPutLabel)
    # TODO: selctor pipeline
    # you can do a grid/random search on the `threshold` parameter for example and so on...
    # SelectorPipeLine = sklearn.pipeline.Pipeline([
    #     ('preprocessing', preprocessing),
    #     ('selector', sklearn.feature_selection.SelectFromModel(sklearn.ensemble.RandomForestRegressor(random_state=42), threshold=0.005)),  # min feature importance
    #     ('svr', sklearn.svm.SVR(C=SVR_RandomSearchCV.best_params_["svr__C"], gamma=SVR_RandomSearchCV.best_params_["svr__gamma"], kernel=SVR_RandomSearchCV.best_params_["svr__kernel"])),
    # ])
    return None

def ClassificationPerformanceMeasures(Setlabels, Predictions, Average : str):
    """
    average='micro': Calculates metrics globally by counting the total true positives, false negatives, and false positives.

    average='macro': Calculates metrics for each label, and finds their unweighted mean (treats all classes equally).

    average='weighted': Like macro, but accounts for class imbalance by weighting each class by its support (the number of true instances).
    """
    ConfusionMatrix = sklearn.metrics.confusion_matrix(Setlabels, Predictions)
    print(ConfusionMatrix)
    Separator()
    PrecisionScore = sklearn.metrics.precision_score(Setlabels, Predictions, average=Average) # ConfusionMatrix[1, 1] / (ConfusionMatrix[0, 1] + ConfusionMatrix[1, 1])
    print(PrecisionScore)
    Separator()
    RecallScore = sklearn.metrics.recall_score(Setlabels, Predictions, average=Average) # ConfusionMatrix[1, 1] / (ConfusionMatrix[1, 0] + ConfusionMatrix[1, 1])
    print(RecallScore)
    Separator()
    F1Score = sklearn.metrics.f1_score(Setlabels, Predictions, average=Average) # ConfusionMatrix[1, 1] / (ConfusionMatrix[1, 1] + (ConfusionMatrix[1, 0] + ConfusionMatrix[0, 1]) / 2)
    print(F1Score)
    Separator()


def MnistExampleClassification(mnist):
    Samples, Labels = mnist.data, mnist.target
    SIZE = len(Samples)
    TestRatio = 0.8
    TestSize : int = int(TestRatio * SIZE)
    TrainSize : int = int(SIZE - TestSize)
    TrainSetsamples, TestSetsamples, TrainSetlabels, TestSetlabels = Samples[:TrainSize], Samples[TestSize:], Labels[:TrainSize], Labels[TestSize:]

    SGDClassifier = sklearn.linear_model.SGDClassifier(random_state=42)

    # accuracies = sklearn.model_selection.cross_val_score(SGDClassifier, TrainSetsamples, TrainSetlabels, cv=2, scoring='accuracy')
    # print(f'cross_val_score : \n{accuracies}')

    Predictions = sklearn.model_selection.cross_val_predict(SGDClassifier, TrainSetsamples, TrainSetlabels, cv=2)
    print(f'cross_val_predict : \n{Predictions}')

    ClassificationPerformanceMeasures(TrainSetlabels, Predictions, 'macro')
    
    return None

InitPlot(RESOURCES_FOLDER_PATH / "images")
# HousingExampleRegression(CSV2DataFrame(RESOURCES_FOLDER_PATH / "ds" / "housing.csv"), "median_house_value")
mnist = sklearn.datasets.fetch_openml('mnist_784', as_frame=False)
MnistExampleClassification(mnist)
