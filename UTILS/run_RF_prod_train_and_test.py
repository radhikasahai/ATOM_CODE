import pyforest
from ATOM_CODE.UTILS.RF_functions import *


if __name__ == '__main__': 
    data_path = '.'
    results_dir='.'
    neks = ['NEK2_binding','NEK2_inhibition','NEK3_binding','NEK5_binding','NEK9_binding','NEK9_inhibition']
    samplings =['none_scaled','UNDER','SMOTE','ADASYN'] 
    feats=['MOE','MFP'] 
    RF_types = ['RF','RF_BCW','BRFC','BRFC_BCW']
    train_results = []
    test_results=[]
    final_cols=['model','NEK','strategy','feat_type','RF_type', 'cm','recall', 'specificity', 'accuracy', 'precision', 
                'f1', 'ROC_AUC', 'MCC', 'balanced_accuracy']
    for nek in neks: 
        for feat in feats: 
            for samp in samplings:
                root_name = f'{nek}_{feat}_{samp}'
                # make sure to remove 
                trainX=pd.read_csv(f'{data_path}{root_name}_trainX.csv')
                train_y=pd.read_csv(f'{data_path}{root_name}_train_y.csv').to_numpy().reshape(-1)
                testX=pd.read_csv(f'{data_path}{root_name}_testX.csv')
                test_y=pd.read_csv(f'{data_path}{root_name}_test_y.csv').to_numpy().reshape(-1)
                for rf in RF_types: 
                    model_name = f'{nek}_{feat}_{samp}_{rf}'
                    print(model_name) 
                    model = rf_models(trainX, train_y, testX, test_y, rf, {}, True)  # make sure dict and doesn't go to default RF version
                    train_df = gather_rf_results(model, trainX, train_y)
                    test_df = gather_rf_results(model, testX, test_y)
                    print()
                    for this_df in [train_df,test_df]: 
                        this_df['model'] = model_name
                        this_df['NEK'] =nek
                        this_df['feat_type'] = feat
                        this_df['strategy'] = samp
                        this_df['RF_type'] = rf
                    
                    train_df.to_csv(f'{results_dir}{model_name}_prod_train.csv',index=False) 
                    test_df.to_csv(f'{results_dir}{model_name}_prod_test.csv',index=False) 

                    train_results.append(train_df.iloc[[0]][final_cols].values.flatten())
                    test_results.append(test_df.iloc[[0]][final_cols].values.flatten())

    all_train =  pd.DataFrame(train_results,columns=final_cols)
    all_test =  pd.DataFrame(test_results,columns=final_cols)
    all_train['modeling_type'] = 'RF' 
    all_train['set'] = 'prod' 
    all_test['modeling_type'] = 'RF' 
    all_test['set'] = 'prod' 
    all_train.to_csv(f'{results_dir}RF_prod_train_results_all_NEK.csv', index=False) 
    all_test.to_csv(f'{results_dir}RF_prod_test_results_all_NEK.csv', index=False)                 
                        
                        
                    

