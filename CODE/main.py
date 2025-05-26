from DataScience import DataScience


process = DataScience()
process.load_dataSet()

process.print_statistical()
process.preprocessing(
    dealing_outlier=True,
    convert_No_Service_to_No=False,
    run_normalize=False,
    selectBestFeatures=True,
    numOfBestFeatures = 10,
    method='label'
)
process.trainModel(
    model='decisionTree',
    useSmoth=False,
    test_size=0.2
)

process.evaluate(
    printResult=True,
    tuneThreshold=True
)

process.showTree(
    max_depth=5,
    fontsize=10
)

process.find_best_parameter_decisionTree(
    showTheGraph=False
)