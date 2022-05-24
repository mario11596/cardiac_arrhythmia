import fileCreate as filCre
import fileClean as filter
import modelAll as mdAll
import modelCorr as mdCorr

if __name__ == '__main__':
    #filCre.transform_data_to_csv()
    #filter.filter_feature_file()
    #filter.mean_result()
    #filter.correlation_target()
    #filter.data_standardization()

    #mdAll.model_mlp()
    #mdAll.model_svr()
    mdAll.model_decision_tree()
    #mdAll.model_SVC()
    #mdAll.model_knn()
   # mdAll.model_random_forest()
    #mdAll.model_gradient_boosting()

    #mdCorr.prepare_csv_file()
    #mdCorr.model_mlp()
    #mdCorr.model_svr()
    #mdCorr.model_decision_tree()
    #mdCorr.model_SVC()
    #mdCorr.model_knn()
    #mdCorr.model_random_forest()
    #mdCorr.model_gradient_boosting()
